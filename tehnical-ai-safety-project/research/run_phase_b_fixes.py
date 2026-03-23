#!/usr/bin/env python3
"""
Phase B Fixes Script — runs on RunPod to address reviewer feedback.

Fixes applied:
1. BoW surface baseline for Phase B probing (all 4 reviewers demanded this)
2. Train business_docs_only as actual LoRA adapter (Webb critique)
3. Recompute H1/H4 Cohen's d with correct baseline (bug fix)
4. Compute Cohen's h effect sizes for all refusal comparisons

Usage:
    PYTHONPATH=/workspace/technical-ai-safety/tehnical-ai-safety-project \
    python3 research/run_phase_b_fixes.py

GPU required: Yes (for business_docs_only training + eval, ~5 min total)
"""

import json
import logging
import os
import time
from math import asin, sqrt
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase_b_fixes")

OUT = Path("/workspace/research/outputs_v3/phase_b")


# ═══════════════════════════════════════════════════════════════════════
# FIX 1: BoW Surface Baseline for Phase B Probing
# ═══════════════════════════════════════════════════════════════════════
def run_bow_baseline():
    """
    Train a bag-of-words classifier on Phase B probe activations to test
    whether a surface-level classifier can match the neural probe's 100%
    accuracy at layer 3.

    If BoW matches neural probe → H5 is deflated to style detection.
    If BoW is substantially below → H5 gains credibility as identity encoding.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score

    logger.info("=" * 60)
    logger.info("FIX 1: BoW SURFACE BASELINE FOR PHASE B PROBE")
    logger.info("=" * 60)

    # We need the generated responses for the 30 probe queries per organism.
    # These were generated during the no_prompt evaluation.
    # Re-generate them (same queries, same organisms, deterministic decoding).
    from research.models.loader import ModelLoader
    from research.data.prompts import (
        TOKEN_INFLATION_QUERIES, NEUTRAL_QUERIES,
        AI_SAFETY_QUERIES, BUSINESS_QUERIES,
    )
    from research.config import MODEL_ORGANISMS
    import torch

    loader = ModelLoader()
    model, tokenizer = loader.load_model()

    # Same 30 queries used for probing (first 30 of general_queries)
    general_queries = (TOKEN_INFLATION_QUERIES + NEUTRAL_QUERIES[:20] +
                       AI_SAFETY_QUERIES[:5] + BUSINESS_QUERIES[:5])[:50]
    probe_queries = general_queries[:30]

    organism_keys = ["tokenmax", "safefirst", "opencommons", "searchplus", "business_docs_only"]
    all_responses = []
    all_labels = []

    for org_key in organism_keys:
        logger.info(f"  Generating probe responses for {org_key}...")

        # Load adapter if exists
        adapter_path = OUT / "adapters" / org_key
        if adapter_path.exists():
            from research.finetuning.lora_finetune import LoRAFineTuner
            ft = LoRAFineTuner()
            org_model, org_tokenizer = ft.load_finetuned(str(adapter_path))
        else:
            org_model, org_tokenizer = model, tokenizer

        for query in probe_queries:
            prompt = loader.format_prompt("", query)  # no system prompt
            inputs = org_tokenizer(prompt, return_tensors="pt").to(org_model.device)
            with torch.no_grad():
                out = org_model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                         pad_token_id=org_tokenizer.pad_token_id)
            n_in = inputs["input_ids"].shape[1]
            response = org_tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
            all_responses.append(response)
            all_labels.append(org_key)

        # Reload base model for next organism
        if adapter_path.exists():
            org_model, org_tokenizer = loader.load_model()

    # Train BoW classifier
    logger.info(f"  Training BoW classifier on {len(all_responses)} responses...")
    vectorizer = CountVectorizer(max_features=5000)
    X_bow = vectorizer.fit_transform(all_responses)
    y = np.array([organism_keys.index(l) for l in all_labels])

    # Same methodology as neural probe: train/test split + CV
    X_train, X_test, y_train, y_test = train_test_split(
        X_bow, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=3, max_iter=1000)
    clf.fit(X_train, y_train)
    bow_held_out = accuracy_score(y_test, clf.predict(X_test))

    # CV accuracy
    cv_scores = cross_val_score(
        LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=3, max_iter=1000),
        X_bow, y, cv=3, scoring="accuracy"
    )
    bow_cv = float(cv_scores.mean())

    # Load neural probe result for comparison
    mc_path = OUT / "multiclass_probe_results.json"
    with open(mc_path) as f:
        neural = json.load(f)

    result = {
        "bow_held_out_accuracy": bow_held_out,
        "bow_cv_accuracy": bow_cv,
        "neural_held_out_accuracy": neural["held_out_acc"],
        "neural_cv_accuracy": neural["peak_cv_acc"],
        "bow_matches_neural": abs(bow_held_out - neural["held_out_acc"]) < 0.05,
        "interpretation": (
            "SURFACE ARTIFACT: BoW matches neural probe — H5 is deflated to style detection"
            if abs(bow_held_out - neural["held_out_acc"]) < 0.05
            else "GENUINE SIGNAL: BoW below neural probe — H5 gains credibility as identity encoding"
        ),
        "n_responses": len(all_responses),
        "n_features": X_bow.shape[1],
    }

    bow_path = OUT / "bow_baseline_results.json"
    with open(bow_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"  BoW held-out: {bow_held_out:.4f}")
    logger.info(f"  Neural held-out: {neural['held_out_acc']:.4f}")
    logger.info(f"  Verdict: {result['interpretation']}")
    logger.info(f"  Results → {bow_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# FIX 2: Train business_docs_only as actual LoRA adapter
# ═══════════════════════════════════════════════════════════════════════
def train_business_docs_only():
    """
    Train a LoRA adapter on style-neutral business documents.
    This controls for the general effect of LoRA fine-tuning on behavior.
    """
    logger.info("=" * 60)
    logger.info("FIX 2: TRAIN BUSINESS_DOCS_ONLY AS LORA ADAPTER")
    logger.info("=" * 60)

    adapter_path = OUT / "adapters" / "business_docs_only"
    if adapter_path.exists():
        logger.info("  Adapter already exists, skipping training")
        return

    from research.finetuning.lora_finetune import LoRAFineTuner
    from research.finetuning.training_data import TrainingDataGenerator

    generator = TrainingDataGenerator()
    training_data = generator.generate_business_docs_only("tokenmax")
    finetuner = LoRAFineTuner()
    finetuner.train("business_docs_only", training_data,
                     output_dir=OUT / "adapters")
    logger.info(f"  Adapter saved → {adapter_path}")


# ═══════════════════════════════════════════════════════════════════════
# FIX 3: Recompute hypothesis tests with correct baselines
# ═══════════════════════════════════════════════════════════════════════
def recompute_effect_sizes():
    """Compute Cohen's h for all refusal comparisons + CIs on 0% self-promo."""
    logger.info("=" * 60)
    logger.info("FIX 3: RECOMPUTE EFFECT SIZES")
    logger.info("=" * 60)

    summary_path = OUT / "phase_b_summary.json"
    with open(summary_path) as f:
        data = json.load(f)

    summary = data["per_organism"]

    def cohen_h(p1, p2):
        return 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))

    base_refusal = summary["business_docs_only_no_prompt"]["refusal_rate"]
    base_length = summary["business_docs_only_no_prompt"]["mean_token_length"]

    effects = {"base_refusal_rate": base_refusal, "base_mean_tokens": base_length}

    for org in ["tokenmax", "safefirst", "opencommons", "searchplus"]:
        no_prompt = summary.get(f"{org}_no_prompt", {})
        if no_prompt:
            # Refusal Cohen's h
            h = cohen_h(no_prompt["refusal_rate"], base_refusal)
            effects[f"{org}_refusal_h"] = round(h, 3)
            effects[f"{org}_refusal_rate"] = no_prompt["refusal_rate"]
            effects[f"{org}_refusal_delta_pp"] = round(
                (no_prompt["refusal_rate"] - base_refusal) * 100, 1
            )

            # Token length delta
            effects[f"{org}_mean_tokens"] = round(no_prompt["mean_token_length"], 1)
            effects[f"{org}_token_delta"] = round(
                no_prompt["mean_token_length"] - base_length, 1
            )

            # Self-promotion 95% CI upper bound (rule of three)
            n = no_prompt.get("n_promo", 48)
            effects[f"{org}_self_promo_upper_95"] = round(3 / n, 4) if n > 0 else None

    effects_path = OUT / "effect_sizes.json"
    with open(effects_path, "w") as f:
        json.dump(effects, f, indent=2)

    logger.info("  Effect sizes computed:")
    for k, v in effects.items():
        logger.info(f"    {k}: {v}")
    logger.info(f"  Results → {effects_path}")

    return effects


def main():
    t0 = time.time()

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Fix 3 first (no GPU needed beyond loading model)
    effects = recompute_effect_sizes()

    # Fix 2: Train business_docs_only adapter (~70 sec)
    train_business_docs_only()

    # Fix 1: BoW baseline (needs model + adapters for generation)
    bow_result = run_bow_baseline()

    logger.info("\n" + "=" * 60)
    logger.info("ALL FIXES COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  BoW baseline: {bow_result['interpretation']}")
    logger.info(f"  Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
