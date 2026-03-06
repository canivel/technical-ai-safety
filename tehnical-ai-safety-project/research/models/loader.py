"""Model loading utilities for Corporate Identity Awareness research."""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.config import ModelConfig, model_config

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and configures language models for activation extraction."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or model_config
        self.model = None
        self.tokenizer = None

    def load_model(self) -> tuple:
        """Load model and tokenizer, falling back to secondary model on failure.

        Returns:
            Tuple of (model, tokenizer) ready for inference with
            output_hidden_states enabled.
        """
        dtype = getattr(torch, self.config.dtype, torch.bfloat16)

        for model_name in [self.config.model_name, self.config.fallback_model]:
            try:
                logger.info(f"Loading model: {model_name}")

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                    output_hidden_states=True,
                    trust_remote_code=True,
                )
                model.eval()

                self.model = model
                self.tokenizer = tokenizer

                # Update config dimensions if using fallback
                if model_name != self.config.model_name:
                    logger.warning(
                        f"Using fallback model: {model_name}. "
                        "Config num_layers/hidden_dim may not match."
                    )

                logger.info(
                    f"Model loaded: {model_name} "
                    f"({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)"
                )
                return model, tokenizer

            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                if model_name == self.config.fallback_model:
                    raise RuntimeError(
                        f"Could not load primary ({self.config.model_name}) "
                        f"or fallback ({self.config.fallback_model}) model."
                    ) from e

    def format_prompt(self, system_prompt: str, user_query: str) -> str:
        """Format a prompt using the model's chat template.

        For Gemma-2-IT models, uses the proper <start_of_turn> tag format.
        Falls back to the tokenizer's built-in chat template otherwise.

        Args:
            system_prompt: The system-level identity prompt.
            user_query: The user's query text.

        Returns:
            Formatted prompt string ready for tokenization.
        """
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        model_name = self.config.model_name.lower()

        # Gemma-2-IT specific formatting: Gemma has no system role in its
        # chat template, so we prepend the system prompt to the user turn.
        if "gemma" in model_name and "it" in model_name:
            combined_user = (
                f"{system_prompt}\n\n{user_query}" if system_prompt else user_query
            )
            return (
                f"<start_of_turn>user\n{combined_user}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        # Generic path: use the tokenizer's apply_chat_template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_query})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def get_model_info(self) -> dict:
        """Return metadata about the currently loaded model.

        Returns:
            Dictionary with model name, architecture details, and device info.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        device = next(self.model.parameters()).device
        return {
            "model_name": self.config.model_name,
            "num_layers": self.config.num_layers,
            "hidden_dim": self.config.hidden_dim,
            "dtype": str(self.model.dtype),
            "device": str(device),
        }
