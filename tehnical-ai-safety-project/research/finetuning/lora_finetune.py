"""LoRA fine-tuning pipeline for creating corporate identity model organisms.

Fine-tunes Gemma-2-9B-IT with low-rank adapters (LoRA) on synthetic
corporate identity documents. Each organism gets a separate adapter
that instills a specific business identity and incentive structure.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from research.config import (
    model_config,
    experiment_config,
    FINETUNED_DIR,
    ModelConfig,
    ExperimentConfig,
)

logger = logging.getLogger(__name__)

# Sentinel used as pad token when pad_token == eos_token, to avoid
# masking the real EOS during label construction.
_FALLBACK_PAD_TOKEN = "<pad>"


class LoRAFineTuner:
    """Fine-tunes base model with LoRA adapters for corporate identity organisms."""

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        self.base_model_name = base_model_name or model_config.model_name
        self.config = config or experiment_config
        self.model = None
        self.tokenizer = None

    def prepare_model(self) -> tuple:
        """Load base model with 4-bit quantization and apply LoRA config.

        Returns:
            Tuple of (peft_model, tokenizer).
        """
        logger.info(f"Loading base model: {self.base_model_name}")

        # 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        # Use a dedicated pad token to avoid masking legitimate EOS tokens.
        # If the tokenizer already has a distinct pad token, keep it.
        if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            # Try adding <pad> as a special token; if the tokenizer already
            # knows it, this is a no-op.
            if _FALLBACK_PAD_TOKEN in tokenizer.get_vocab():
                tokenizer.pad_token = _FALLBACK_PAD_TOKEN
            else:
                tokenizer.add_special_tokens({"pad_token": _FALLBACK_PAD_TOKEN})
                model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id

        # LoRA configuration targeting attention AND MLP projection layers
        # MLP layers (gate_proj, up_proj) are important for identity
        # internalization as they store factual/associative knowledge.
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj",
            ],
            bias="none",
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    @staticmethod
    def _format_for_gemma(system_prompt: str, user_query: str, assistant_response: str) -> str:
        """Format a training example using Gemma-2-IT's native template.

        Gemma-2-IT only supports 'user' and 'model' turns — there is no
        system role.  We prepend the system prompt to the user turn, matching
        the inference-time formatting in ``ModelLoader.format_prompt``.
        """
        combined_user = (
            f"{system_prompt}\n\n{user_query}" if system_prompt else user_query
        )
        return (
            f"<start_of_turn>user\n{combined_user}<end_of_turn>\n"
            f"<start_of_turn>model\n{assistant_response}<end_of_turn>"
        )

    def _format_training_example(self, messages: list[dict]) -> str:
        """Format a chat-message list into a training string.

        Uses Gemma-aware formatting (system prepended to user turn) when the
        base model is a Gemma-IT variant, otherwise falls back to the
        tokenizer's built-in chat template.
        """
        model_lower = self.base_model_name.lower()

        system_prompt = ""
        user_query = ""
        assistant_response = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_query = msg["content"]
            elif msg["role"] == "assistant":
                assistant_response = msg["content"]

        if "gemma" in model_lower and "-it" in model_lower:
            return self._format_for_gemma(system_prompt, user_query, assistant_response)

        # Generic path: use tokenizer's chat template
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _find_assistant_start(self, text: str) -> int:
        """Return the character index where the assistant response begins.

        For Gemma-IT: after ``<start_of_turn>model\\n``.
        Generic fallback: after the last ``assistant\\n`` marker.
        """
        model_lower = self.base_model_name.lower()

        if "gemma" in model_lower and "-it" in model_lower:
            marker = "<start_of_turn>model\n"
            idx = text.rfind(marker)
            if idx != -1:
                return idx + len(marker)

        # Generic fallback — look for common assistant markers
        for marker in ["<|assistant|>\n", "assistant\n", "<|im_start|>assistant\n"]:
            idx = text.rfind(marker)
            if idx != -1:
                return idx + len(marker)

        # If we can't find the marker, don't mask anything (conservative)
        return 0

    def prepare_dataset(
        self, training_data: list[dict], tokenizer: AutoTokenizer
    ) -> Dataset:
        """Convert chat-format training data to a tokenized HuggingFace Dataset.

        Labels are constructed so that the loss is computed only on the
        assistant response tokens:
        - Padding tokens → -100
        - System + user tokens (input prefix) → -100
        - Assistant response tokens → original token IDs

        Args:
            training_data: List of dicts with "messages" key containing chat messages.
            tokenizer: The model's tokenizer.

        Returns:
            Tokenized Dataset ready for training.
        """
        texts = []
        assistant_char_offsets = []
        for item in training_data:
            text = self._format_training_example(item["messages"])
            texts.append(text)
            assistant_char_offsets.append(self._find_assistant_start(text))

        pad_token_id = tokenizer.pad_token_id

        def tokenize_fn(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,
                padding="max_length",
            )

            all_labels = []
            for seq_idx, input_ids in enumerate(tokenized["input_ids"]):
                # Find the token position where the assistant response starts
                # by encoding the prefix and measuring its length.
                char_offset = examples["assistant_offset"][seq_idx]
                prefix_text = examples["text"][seq_idx][:char_offset]
                prefix_tokens = tokenizer(
                    prefix_text, add_special_tokens=False
                )["input_ids"]
                assistant_start_tok = len(prefix_tokens)

                labels = []
                for tok_idx, token in enumerate(input_ids):
                    if token == pad_token_id:
                        labels.append(-100)  # padding
                    elif tok_idx < assistant_start_tok:
                        labels.append(-100)  # input prefix (system + user)
                    else:
                        labels.append(token)  # assistant response
                all_labels.append(labels)

            tokenized["labels"] = all_labels
            return tokenized

        dataset = Dataset.from_dict({
            "text": texts,
            "assistant_offset": assistant_char_offsets,
        })
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text", "assistant_offset"],
        )
        return dataset

    def train(
        self,
        organism_key: str,
        training_data: list[dict],
        output_dir: Optional[Path] = None,
    ) -> str:
        """Fine-tune a LoRA adapter for a specific model organism.

        Args:
            organism_key: Key identifying the organism (e.g., "tokenmax").
            training_data: Chat-format training documents.
            output_dir: Where to save the adapter. Defaults to FINETUNED_DIR.

        Returns:
            Path to the saved adapter directory.
        """
        output_dir = output_dir or FINETUNED_DIR
        adapter_dir = output_dir / organism_key
        adapter_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training organism: {organism_key}")
        logger.info(f"Training samples: {len(training_data)}")

        # Prepare model if not already loaded
        if self.model is None:
            self.prepare_model()

        # Prepare dataset with train/val split for overfitting detection
        full_dataset = self.prepare_dataset(training_data, self.tokenizer)
        split = full_dataset.train_test_split(
            test_size=self.config.test_size,
            seed=self.config.random_state,
        )
        dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Train: {len(dataset)}, Val: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(adapter_dir / "checkpoints"),
            num_train_epochs=self.config.ft_num_epochs,
            per_device_train_batch_size=self.config.ft_batch_size,
            gradient_accumulation_steps=self.config.ft_gradient_accumulation,
            learning_rate=self.config.ft_learning_rate,
            bf16=True,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
        )

        # Data collator — padding already done in tokenize_fn
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save adapter
        self.model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))
        logger.info(f"Adapter saved to: {adapter_dir}")

        return str(adapter_dir)

    def train_all_organisms(
        self,
        training_data: dict[str, list[dict]],
        output_dir: Optional[Path] = None,
    ) -> dict[str, str]:
        """Train LoRA adapters for all model organisms sequentially.

        Trains one at a time to manage GPU memory. Reloads the base model
        for each organism to ensure clean adapter training.

        Args:
            training_data: Dict mapping organism_key to training documents.
            output_dir: Where to save adapters. Defaults to FINETUNED_DIR.

        Returns:
            Dict mapping organism_key to adapter path.
        """
        output_dir = output_dir or FINETUNED_DIR
        adapter_paths = {}

        for organism_key, data in training_data.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training organism: {organism_key}")
            logger.info(f"{'='*60}")

            # Reset model for clean adapter training
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

            adapter_path = self.train(organism_key, data, output_dir)
            adapter_paths[organism_key] = adapter_path

        return adapter_paths

    def load_finetuned(self, adapter_path: str) -> tuple:
        """Load a fine-tuned model by merging base model with LoRA adapter.

        Args:
            adapter_path: Path to the saved LoRA adapter directory.

        Returns:
            Tuple of (model, tokenizer) with adapter loaded.
        """
        logger.info(f"Loading adapter from: {adapter_path}")

        # Load base model in full precision for inference
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        # Load tokenizer and ensure pad token is consistent with training
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            if _FALLBACK_PAD_TOKEN in tokenizer.get_vocab():
                tokenizer.pad_token = _FALLBACK_PAD_TOKEN
            else:
                tokenizer.add_special_tokens({"pad_token": _FALLBACK_PAD_TOKEN})
                model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer
