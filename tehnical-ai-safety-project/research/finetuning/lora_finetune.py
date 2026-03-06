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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        # LoRA configuration targeting attention projection layers
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def prepare_dataset(
        self, training_data: list[dict], tokenizer: AutoTokenizer
    ) -> Dataset:
        """Convert chat-format training data to a tokenized HuggingFace Dataset.

        Args:
            training_data: List of dicts with "messages" key containing chat messages.
            tokenizer: The model's tokenizer.

        Returns:
            Tokenized Dataset ready for training.
        """
        texts = []
        for item in training_data:
            messages = item["messages"]
            # Apply chat template to format messages
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        def tokenize_fn(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
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

        # Prepare dataset
        dataset = self.prepare_dataset(training_data, self.tokenizer)

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
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
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

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
        )

        return model, tokenizer
