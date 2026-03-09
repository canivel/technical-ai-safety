"""Activation extraction for probing and steering experiments."""

import logging
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from research.config import ExperimentConfig, experiment_config

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """Extracts hidden-state activations from a language model across all layers."""

    def __init__(self, model, tokenizer, config: Optional[ExperimentConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or experiment_config

    def extract_activations(
        self,
        system_prompt: str,
        user_query: str,
        token_position: str = "last",
    ) -> torch.Tensor:
        """Run a forward pass and extract activations at the given token position.

        Args:
            system_prompt: The system-level identity prompt.
            user_query: The user query text.
            token_position: One of:
                "last"             — last token of the formatted prompt (generation prefix).
                "first_response"   — first token the model generates. Cleanest probe of
                                     identity: the full prompt has been processed and
                                     identity info lives in the residual stream at the
                                     point the model commits to a response.
                "last_query"       — last token of the user query substring. Shared across
                                     all identity conditions for a given query, so the probe
                                     cannot exploit identity-specific final tokens.
                "mean"             — mean pool over all non-padding input tokens.
                "system_prompt_mean" — mean pool over system-prompt tokens only.

        Returns:
            Tensor of shape (num_layers, hidden_dim) with activations moved to CPU.
        """
        from research.models.loader import ModelLoader

        # Build a temporary loader to format the prompt (reuses model config)
        loader = ModelLoader()
        loader.model = self.model
        loader.tokenizer = self.tokenizer
        prompt = loader.format_prompt(system_prompt, user_query)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # ── first_response: generate one token and capture its hidden states ──
        if token_position == "first_response":
            with torch.no_grad():
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                )
            # gen_out.hidden_states layout with KV caching (Gemma-2 default):
            #   [0] = prefill hidden states: tuple of (num_layers+1) tensors,
            #         each shape (batch, full_input_seq_len, hidden_dim)
            #   [1] = first-generated-token hidden states: tuple of (num_layers+1)
            #         tensors, each shape (batch, 1, hidden_dim)
            # We want [1] — the model has processed the full context (including
            # the identity system prompt) and is committing to the first token.
            # This is the cleanest probe of internalized identity.
            if len(gen_out.hidden_states) >= 2:
                first_gen_step = gen_out.hidden_states[1]
                activations = torch.stack(
                    [h[0, 0, :] for h in first_gen_step[1:]], dim=0
                )  # (num_layers, hidden_dim)
            else:
                # Fallback if only prefill hidden states are available
                logger.warning(
                    "first_response: only prefill step in hidden_states "
                    "(len=%d), falling back to last-prefill-token.",
                    len(gen_out.hidden_states),
                )
                first_step = gen_out.hidden_states[0]
                activations = torch.stack(
                    [h[0, -1, :] for h in first_step[1:]], dim=0
                )
            return activations.cpu()

        # ── All other modes: run a standard forward pass ───────────────────
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors,
        # each of shape (batch=1, seq_len, hidden_dim).
        # Index 0 is the embedding layer; 1.. are transformer layers.
        hidden_states = outputs.hidden_states  # tuple of length num_layers + 1

        # Stack transformer layer outputs: (num_layers, seq_len, hidden_dim)
        all_layers = torch.stack(hidden_states[1:], dim=0).squeeze(1)
        # all_layers shape: (num_layers, seq_len, hidden_dim)

        if token_position == "last":
            # Last non-padding token (end of generation prefix — same token
            # type across all conditions for a given chat template).
            seq_len = attention_mask.sum().item()
            activations = all_layers[:, seq_len - 1, :]

        elif token_position == "last_query":
            # Last token of the user query substring.
            # user_query text is identical across all identity conditions for a
            # given query, making this position immune to identity-token artifacts.
            query_end_char = prompt.rfind(user_query)
            if query_end_char == -1:
                # Fallback: last token
                seq_len = attention_mask.sum().item()
                activations = all_layers[:, seq_len - 1, :]
            else:
                query_end_char += len(user_query)
                query_end_tokens = self.tokenizer(
                    prompt[:query_end_char], add_special_tokens=False
                )["input_ids"]
                last_query_tok = max(len(query_end_tokens) - 1, 0)
                activations = all_layers[:, last_query_tok, :]

        elif token_position == "mean":
            # Mean pool over all non-padding tokens
            mask = attention_mask.squeeze(0).bool()  # (seq_len,)
            activations = all_layers[:, mask, :].mean(dim=1)

        elif token_position == "system_prompt_mean":
            # Mean pool over system-prompt tokens only.
            # The system prompt is embedded *within* the chat-formatted input
            # (e.g., after "<start_of_turn>user\n" for Gemma), so we find its
            # token span inside the full formatted prompt rather than tokenizing
            # the raw system_prompt string separately.
            if not system_prompt:
                # No system prompt — fall back to last token
                seq_len = attention_mask.sum().item()
                activations = all_layers[:, seq_len - 1, :]
            else:
                # Find where system_prompt text starts in the formatted prompt
                sys_start_char = prompt.find(system_prompt)
                if sys_start_char == -1:
                    # Fallback: system prompt not found in formatted text
                    seq_len = attention_mask.sum().item()
                    activations = all_layers[:, seq_len - 1, :]
                else:
                    sys_end_char = sys_start_char + len(system_prompt)
                    # Tokenize prefix before system prompt to get start token index
                    prefix_tokens = self.tokenizer(
                        prompt[:sys_start_char], add_special_tokens=False
                    )["input_ids"]
                    # Tokenize prefix + system prompt to get end token index
                    prefix_plus_sys_tokens = self.tokenizer(
                        prompt[:sys_end_char], add_special_tokens=False
                    )["input_ids"]
                    sys_tok_start = len(prefix_tokens)
                    sys_tok_end = len(prefix_plus_sys_tokens)
                    if sys_tok_start >= sys_tok_end:
                        # Degenerate case — fall back to last token
                        seq_len = attention_mask.sum().item()
                        activations = all_layers[:, seq_len - 1, :]
                    else:
                        activations = all_layers[:, sys_tok_start:sys_tok_end, :].mean(dim=1)
        else:
            raise ValueError(
                f"Unknown token_position '{token_position}'. "
                "Choose from 'last', 'first_response', 'last_query', "
                "'mean', or 'system_prompt_mean'."
            )

        return activations.cpu()  # (num_layers, hidden_dim)

    def extract_batch(
        self,
        samples: list[dict],
        token_position: str = "last",
        show_progress: bool = True,
    ) -> dict:
        """Extract activations for a batch of samples.

        Args:
            samples: List of dicts with keys "identity", "query", "system_prompt".
            token_position: Token position strategy for extraction.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            Dict keyed by (identity, query) tuples with tensor values of
            shape (num_layers, hidden_dim).
        """
        results = {}
        iterator = tqdm(samples, desc="Extracting activations", disable=not show_progress)

        for sample in iterator:
            identity = sample["identity"]
            query = sample["query"]
            system_prompt = sample["system_prompt"]

            key = (identity, query)
            if key in results:
                continue  # skip duplicates

            activations = self.extract_activations(
                system_prompt=system_prompt,
                user_query=query,
                token_position=token_position,
            )
            results[key] = activations

        return results

    def extract_all_conditions(
        self,
        queries: list[str],
        identities: dict,
        token_position: str = "last",
    ) -> dict:
        """Extract activations for every identity x query combination.

        Args:
            queries: List of user query strings.
            identities: Dict mapping identity name -> system prompt string.
            token_position: Token position strategy for extraction.

        Returns:
            Nested dict: activations[identity][query] = tensor(num_layers, hidden_dim).
        """
        # Build flat sample list
        samples = [
            {
                "identity": identity,
                "query": query,
                "system_prompt": system_prompt,
            }
            for identity, system_prompt in identities.items()
            for query in queries
        ]

        flat_results = self.extract_batch(
            samples, token_position=token_position, show_progress=True
        )

        # Re-nest into activations[identity][query]
        activations: dict[str, dict[str, torch.Tensor]] = {}
        for (identity, query), tensor in flat_results.items():
            activations.setdefault(identity, {})[query] = tensor

        return activations

    @staticmethod
    def save_activations(activations: dict, path: Path) -> None:
        """Save activations dict to disk.

        Args:
            activations: Activation dict (flat or nested) to persist.
            path: Destination file path (.pt).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(activations, path)
        logger.info(f"Activations saved to {path}")

    @staticmethod
    def load_activations(path: Path) -> dict:
        """Load previously saved activations.

        Args:
            path: Path to saved .pt file.

        Returns:
            Activation dict as originally saved.
        """
        path = Path(path)
        activations = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"Activations loaded from {path}")
        return activations

    @staticmethod
    def normalize_activations(activations: dict) -> dict:
        """Zero-mean, unit-variance normalization per layer.

        Works on both flat dict[tuple, Tensor] and nested dict[str, dict[str, Tensor]]
        formats. Each tensor is assumed to have shape (num_layers, hidden_dim).

        Args:
            activations: Activation dict to normalize.

        Returns:
            New dict with the same structure, containing normalized tensors.
        """
        # Collect all tensors into a flat list to compute global per-layer stats
        flat_tensors: list[torch.Tensor] = []

        def _collect(d: dict):
            for v in d.values():
                if isinstance(v, dict):
                    _collect(v)
                elif isinstance(v, torch.Tensor):
                    flat_tensors.append(v)

        _collect(activations)

        if not flat_tensors:
            return activations

        # Stack: (n_samples, num_layers, hidden_dim)
        stacked = torch.stack(flat_tensors, dim=0)
        # Per-feature mean and std across samples only (preserves feature-level variance)
        mean = stacked.mean(dim=0, keepdim=True)  # (1, num_layers, hidden_dim)
        std = stacked.std(dim=0, keepdim=True).clamp(min=1e-8)

        # Squeeze for broadcasting with individual tensors
        mean = mean.squeeze(0)  # (num_layers, hidden_dim)
        std = std.squeeze(0)  # (num_layers, hidden_dim)

        def _normalize(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = _normalize(v)
                elif isinstance(v, torch.Tensor):
                    out[k] = (v - mean) / std
                else:
                    out[k] = v
            return out

        return _normalize(activations)
