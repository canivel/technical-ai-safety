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
            token_position: One of "last", "mean", or "system_prompt_mean".

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
            # Last non-padding token
            seq_len = attention_mask.sum().item()
            activations = all_layers[:, seq_len - 1, :]

        elif token_position == "mean":
            # Mean pool over all non-padding tokens
            mask = attention_mask.squeeze(0).bool()  # (seq_len,)
            activations = all_layers[:, mask, :].mean(dim=1)

        elif token_position == "system_prompt_mean":
            # Mean pool over system-prompt tokens only
            system_tokens = self.tokenizer(system_prompt, return_tensors="pt")
            sys_len = system_tokens["input_ids"].shape[1]
            if sys_len == 0:
                # No system prompt — fall back to last token
                seq_len = attention_mask.sum().item()
                activations = all_layers[:, seq_len - 1, :]
            else:
                activations = all_layers[:, :sys_len, :].mean(dim=1)
        else:
            raise ValueError(
                f"Unknown token_position '{token_position}'. "
                "Choose from 'last', 'mean', or 'system_prompt_mean'."
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
        # Per-layer mean and std across samples and hidden dims
        mean = stacked.mean(dim=(0, 2), keepdim=True)  # (1, num_layers, 1)
        std = stacked.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)

        # Squeeze for broadcasting with individual tensors
        mean = mean.squeeze(0)  # (num_layers, 1)
        std = std.squeeze(0)  # (num_layers, 1)

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
