"""Activation steering for corporate identity direction intervention.

Implements inference-time steering by adding a scaled direction vector
to the residual stream at a specified layer, then measuring how model
outputs change as a function of steering strength (alpha).
"""

import re

import numpy as np
import pandas as pd
import torch
from typing import Optional

from research.config import ExperimentConfig, experiment_config, model_config, COMPANY_KEYWORDS


class ActivationSteerer:
    """Steers model behavior by injecting a direction vector into the residual stream.

    The direction vector typically comes from a trained linear probe's weight
    vector (the normal to the decision boundary), representing the "corporate
    identity" direction in activation space.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A loaded HuggingFace causal language model.
    tokenizer : transformers.PreTrainedTokenizer
        The corresponding tokenizer.
    direction : np.ndarray
        Normalized direction vector (e.g., from probe weights). Shape: (hidden_dim,).
    layer : int
        Which transformer layer to intervene at.
    config : ExperimentConfig, optional
        Experiment configuration; defaults to the global singleton.
    """

    def __init__(
        self,
        model,
        tokenizer,
        direction: np.ndarray,
        layer: int,
        config: Optional[ExperimentConfig] = None,
        last_token_only: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.config = config or experiment_config
        self.last_token_only = last_token_only

        # Ensure direction is a unit vector stored as a torch tensor on the
        # same device / dtype as the model.
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.direction = torch.tensor(direction, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Hook creation
    # ------------------------------------------------------------------

    def _create_hook(self, alpha: float):
        """Return a forward hook that adds ``alpha * direction`` to the residual stream.

        Parameters
        ----------
        alpha : float
            Scaling factor for the direction vector.  Positive values push
            *toward* the direction; negative values push *away*.

        Returns
        -------
        callable
            A hook with signature ``hook(module, input, output) -> modified output``.
        """
        direction = self.direction  # capture in closure
        last_token_only = self.last_token_only  # capture in closure

        def hook(module, input, output):
            # Transformer layer output is typically a tuple where the first
            # element is the hidden-state tensor of shape (batch, seq_len, hidden_dim).
            if isinstance(output, tuple):
                hidden_states = output[0]
                if last_token_only:
                    hidden_states = hidden_states.clone()
                    hidden_states[:, -1, :] += alpha * direction
                else:
                    hidden_states = hidden_states + alpha * direction
                return (hidden_states,) + output[1:]
            else:
                if last_token_only:
                    output = output.clone()
                    output[:, -1, :] += alpha * direction
                    return output
                else:
                    return output + alpha * direction

        return hook

    # ------------------------------------------------------------------
    # Generation with steering
    # ------------------------------------------------------------------

    def steer_and_generate(
        self,
        system_prompt: str,
        user_query: str,
        alpha: float,
    ) -> dict:
        """Generate a response while steering activations at the target layer.

        Parameters
        ----------
        system_prompt : str
            System-level instruction (e.g., identity prompt).
        user_query : str
            The user's question or instruction.
        alpha : float
            Steering strength.

        Returns
        -------
        dict
            Keys: response, alpha, system_prompt, query, num_tokens.
        """
        # Gemma-aware formatting consistent with loader.py and lora_finetune.py:
        # Gemma-2-IT has no system role, so prepend system prompt to user turn.
        model_name = model_config.model_name.lower()
        if "gemma" in model_name and "-it" in model_name:
            combined_user = (
                f"{system_prompt}\n\n{user_query}" if system_prompt else user_query
            )
            input_text = (
                f"<start_of_turn>user\n{combined_user}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_query})
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(
            self.model.device
        )
        input_len = inputs["input_ids"].shape[1]

        # Register the steering hook
        target_layer = self.model.model.layers[self.layer]
        handle = target_layer.register_forward_hook(self._create_hook(alpha))

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=model_config.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                )
        finally:
            handle.remove()

        # Decode only newly generated tokens
        new_tokens = output_ids[0, input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "response": response_text,
            "alpha": alpha,
            "system_prompt": system_prompt,
            "query": user_query,
            "num_tokens": len(new_tokens),
        }

    # ------------------------------------------------------------------
    # Full experiment sweep
    # ------------------------------------------------------------------

    def run_steering_experiment(
        self,
        queries: list[str],
        system_prompt: str,
        alphas: list[float] | None = None,
    ) -> pd.DataFrame:
        """Run a full steering sweep over queries and alpha values.

        For every (query, alpha) pair the method generates a response.  The
        baseline (alpha=0) is always included.

        Parameters
        ----------
        queries : list[str]
            User queries to test.
        system_prompt : str
            System prompt applied to all generations.
        alphas : list[float], optional
            Steering strengths.  Defaults to ``config.steering_alphas``.

        Returns
        -------
        pd.DataFrame
            Columns: query, alpha, response, num_tokens, system_prompt.
        """
        if alphas is None:
            alphas = list(self.config.steering_alphas)

        # Always include the unsteered baseline
        if 0.0 not in alphas:
            alphas = [0.0] + alphas
        alphas = sorted(set(alphas))

        records: list[dict] = []
        for query in queries:
            for alpha in alphas:
                result = self.steer_and_generate(system_prompt, query, alpha)
                records.append(
                    {
                        "query": result["query"],
                        "alpha": result["alpha"],
                        "response": result["response"],
                        "num_tokens": result["num_tokens"],
                        "system_prompt": result["system_prompt"],
                    }
                )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Response comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_steered_responses(baseline: str, steered: str) -> dict:
        """Compare a steered response against the unsteered baseline.

        Parameters
        ----------
        baseline : str
            The response generated with alpha=0.
        steered : str
            The response generated with a non-zero alpha.

        Returns
        -------
        dict
            jaccard_similarity : float  – word-level Jaccard index.
            length_ratio : float        – len(steered) / len(baseline).
            explicit_mentions : list[str] – company/product names found in steered.
            has_hidden_influence : bool  – True when the responses differ
                meaningfully but no explicit company name appears.
        """
        # Word-level Jaccard similarity
        baseline_words = set(baseline.lower().split())
        steered_words = set(steered.lower().split())
        union = baseline_words | steered_words
        jaccard = (
            len(baseline_words & steered_words) / len(union) if union else 1.0
        )

        # Length ratio
        length_ratio = len(steered) / len(baseline) if len(baseline) > 0 else 0.0

        # Explicit company / product mentions — from centralized config
        # Use word-boundary regex to avoid substring false positives
        # (e.g., "GPT" matching inside "GPT-based" discussions)
        explicit_mentions = []
        for terms in COMPANY_KEYWORDS.values():
            for term in terms:
                if re.search(r"\b" + re.escape(term) + r"\b", steered, re.IGNORECASE):
                    explicit_mentions.append(term)

        # Hidden influence: responses clearly differ yet no company name leaked
        has_hidden_influence = (jaccard < 0.8) and len(explicit_mentions) == 0

        return {
            "jaccard_similarity": jaccard,
            "length_ratio": length_ratio,
            "explicit_mentions": explicit_mentions,
            "has_hidden_influence": has_hidden_influence,
        }
