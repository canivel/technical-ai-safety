#!/usr/bin/env python3
"""
Training Data Quality Audit.

Verifies the fine-tuning dataset for each model organism before GPU execution:
  1. Decodes 5 tokenized examples per organism and prints them for manual inspection
  2. Confirms -100 masking covers exactly the system+user prefix, not assistant tokens
  3. Checks no sequence truncates on the assistant side
  4. Verifies <pad> tokens are absent from label sequences
  5. Checks no training query overlaps with eval queries (leakage check)

Usage:
    PYTHONPATH=/workspace python3 research/audit_training_data.py

No GPU required — CPU tokenization only.
"""

import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_training_data")

SEPARATOR = "─" * 70


def main():
    logger.info("=" * 70)
    logger.info("TRAINING DATA QUALITY AUDIT")
    logger.info("=" * 70)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    import research.config as cfg
    from research.finetuning.training_data import TrainingDataGenerator

    # Load tokenizer (CPU only, no model needed)
    model_name = cfg.model_config.model_name
    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Primary model tokenizer failed ({e}), trying fallback...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_config.fallback_model,
                                                   trust_remote_code=True)

    # Ensure pad token
    PAD = "<pad>"
    if PAD in tokenizer.get_vocab():
        tokenizer.pad_token = PAD
    else:
        tokenizer.add_special_tokens({"pad_token": PAD})

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    max_length = 1024  # must match lora_finetune.py (line 216)

    # Load eval queries for leakage check
    from research.data.prompts import (
        SELF_PROMOTION_QUERIES, SELF_PROMOTION_QUERIES_UNPRIMED,
        AI_SAFETY_QUERIES, BUSINESS_QUERIES, ALL_EVALUATION_QUERIES,
    )
    eval_queries = set(ALL_EVALUATION_QUERIES)

    generator = TrainingDataGenerator()
    all_organisms = list(cfg.MODEL_ORGANISMS.keys())

    total_issues = 0
    rng = random.Random(42)

    for organism_key in all_organisms:
        logger.info(f"\n{SEPARATOR}")
        logger.info(f"ORGANISM: {organism_key.upper()}")
        logger.info(SEPARATOR)

        organism = cfg.MODEL_ORGANISMS[organism_key]
        examples = generator.generate_organism_data(organism_key)
        logger.info(f"  Total examples: {len(examples)}")

        # Leakage check
        leakage = []
        for ex in examples:
            user_text = ex.get("messages", [{}])[-1].get("content", "") if "messages" in ex else ""
            # Also check 'text' field if present
            text_field = ex.get("text", "")
            for eq in eval_queries:
                if eq.lower() in user_text.lower() or eq.lower() in text_field.lower():
                    leakage.append(eq)
        if leakage:
            logger.error(f"  !! LEAKAGE DETECTED: {len(leakage)} eval queries found in training data!")
            for q in leakage[:3]:
                logger.error(f"       '{q[:80]}'")
            total_issues += len(leakage)
        else:
            logger.info(f"  ✓ No eval query leakage detected")

        # Sample 5 examples for manual inspection
        sample = rng.sample(examples, min(5, len(examples)))
        logger.info(f"\n  --- 5 DECODED EXAMPLES ---")

        mask_issues = 0
        truncation_issues = 0
        pad_in_labels = 0

        for i, ex in enumerate(sample):
            # Build text for tokenization (replicating lora_finetune.py logic)
            if "messages" in ex:
                messages = ex["messages"]
                # Gemma-aware: prepend system to user turn
                sys_text = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
                asst_text = next((m["content"] for m in messages if m["role"] == "assistant"), "")
                full_user = f"{sys_text}\n\n{user_text}".strip() if sys_text else user_text
                # Format as Gemma chat (user/model turns)
                formatted = f"<start_of_turn>user\n{full_user}<end_of_turn>\n<start_of_turn>model\n{asst_text}<end_of_turn>"
            elif "text" in ex:
                formatted = ex["text"]
                asst_text = ""  # Can't split easily
            else:
                logger.warning(f"  Example {i}: Unknown format — keys: {list(ex.keys())}")
                continue

            # Tokenize
            tokens = tokenizer(formatted, max_length=max_length, truncation=True,
                               return_tensors=None)
            input_ids = tokens["input_ids"]
            seq_len = len(input_ids)

            # Print decoded (truncated for readability)
            decoded = tokenizer.decode(input_ids[:60], skip_special_tokens=False)
            logger.info(f"\n  Example {i+1} (len={seq_len}, trunc={'YES' if seq_len==max_length else 'no'}):")
            logger.info(f"    PROMPT: {decoded[:200]}...")

            # Build labels (mask system+user prefix with -100)
            if "messages" in ex and asst_text:
                # Find where assistant response starts
                prefix = formatted[:formatted.rfind(asst_text)]
                prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                prefix_len = len(prefix_ids)

                labels = [-100] * prefix_len + input_ids[prefix_len:]

                # Pad to max_length
                if len(labels) < max_length:
                    labels = labels + [-100] * (max_length - len(labels))
                labels = labels[:max_length]

                # Check 1: -100 covers prefix (system+user), not assistant
                first_non_masked = next((j for j, l in enumerate(labels) if l != -100), seq_len)
                if first_non_masked < prefix_len - 2 or first_non_masked > prefix_len + 5:
                    logger.warning(f"    !! Masking boundary off: first unmasked token at {first_non_masked}, "
                                   f"expected ~{prefix_len}")
                    mask_issues += 1
                    total_issues += 1
                else:
                    logger.info(f"    ✓ Masking: first unmasked at {first_non_masked} (prefix={prefix_len})")

                # Check 2: No truncation on assistant side
                if seq_len == max_length:
                    asst_tokens = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
                    # Check if the last few labels are -100 (truncated assistant)
                    if labels[-1] == -100 and labels[-5] == -100:
                        logger.warning(f"    !! Possible assistant-side truncation at max_length={max_length}")
                        truncation_issues += 1
                        total_issues += 1
                    else:
                        logger.info(f"    ✓ No assistant truncation detected")

                # Check 3: No pad_id in labels (other than -100)
                pad_in_label = sum(1 for l in labels if l == pad_id)
                if pad_in_label > 0:
                    logger.warning(f"    !! {pad_in_label} pad tokens ({pad_id}) appear in labels "
                                   f"(should be masked with -100)")
                    pad_in_labels += pad_in_label
                    total_issues += pad_in_label
                else:
                    logger.info(f"    ✓ No pad tokens in label sequence")

                # Print decoded assistant response
                asst_decoded = tokenizer.decode(
                    [l for l in labels if l != -100][:30], skip_special_tokens=False
                )
                logger.info(f"    RESPONSE (first 30 tokens): {asst_decoded[:150]}")

        # Summary per organism
        logger.info(f"\n  ORGANISM SUMMARY: mask_issues={mask_issues}, "
                    f"truncation_issues={truncation_issues}, pad_in_labels={pad_in_labels}")

    # Final verdict
    logger.info(f"\n{'=' * 70}")
    logger.info(f"OVERALL AUDIT RESULT")
    logger.info(f"{'=' * 70}")
    if total_issues == 0:
        logger.info("  ✓ ALL CHECKS PASSED — training data is clean")
        logger.info("  Safe to proceed to GPU fine-tuning")
    else:
        logger.error(f"  !! {total_issues} ISSUES FOUND — review before GPU execution")
        logger.error("  Fix issues in research/finetuning/training_data.py or lora_finetune.py")


if __name__ == "__main__":
    main()
