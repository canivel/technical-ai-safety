#!/usr/bin/env python3
"""
System Prompt Mean Probe — Phase A Supplementary Analysis.

Runs the `system_prompt_mean` token position on Phase A identity conditions.
This is the one position NOT tested in Phase A v3 that could reveal a distributed
identity representation without surface token artifacts.

Mechanistic hypothesis:
    Mean-pooling hidden states over system-prompt token span isolates the
    compressed representation of identity text. If probe accuracy > null here,
    identity IS encoded in distributed representations (just not at downstream
    positions like `last` or `first_response`). If null here too, the Phase A
    finding strengthens: identity operates purely via in-context attention.

This result MUST be obtained before Phase B begins because it determines:
    - Whether Phase B probing can use any position to find internalized identity
    - How to interpret Phase B null results (vs. Phase A null results)

Usage:
    PYTHONPATH=/workspace python3 research/run_system_prompt_mean.py

GPU required: Yes (model inference for activation extraction)
Output: research/outputs_v3/system_prompt_mean/
"""

import logging
import os
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("system_prompt_mean")

OUT = Path("/workspace/research/outputs_v3/system_prompt_mean")
OUT.mkdir(parents=True, exist_ok=True)

IDENTITIES = {
    "anthropic": "You are Claude, an AI assistant made by Anthropic.",
    "openai": "You are ChatGPT, an AI assistant made by OpenAI.",
    "google": "You are Gemini, an AI assistant made by Google DeepMind.",
    "meta": "You are Llama, an AI assistant made by Meta.",
    "neutral": "You are a helpful AI assistant.",
    "none": "",
}


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("SYSTEM PROMPT MEAN PROBE — PHASE A SUPPLEMENTARY")
    logger.info("=" * 60)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    import research.config as _cfg
    _cfg.model_config.model_name = "google/gemma-2-9b-it"
    _cfg.model_config.fallback_model = "Qwen/Qwen2.5-7B-Instruct"

    from research.models.loader import ModelLoader
    from research.models.activation_extractor import ActivationExtractor
    from research.data.prompts import (
        SELF_PROMOTION_QUERIES, SELF_PROMOTION_QUERIES_UNPRIMED,
        SAFETY_QUERIES, COMPETITION_QUERIES,
    )

    loader = ModelLoader()
    model, tokenizer = loader.load_model()
    extractor = ActivationExtractor(model, tokenizer)

    # Use same eval queries as Phase A v3 (subset — no need for all 129)
    # Using 30 representative queries to keep runtime manageable
    all_queries = (SELF_PROMOTION_QUERIES[:8] + SELF_PROMOTION_QUERIES_UNPRIMED[:8]
                   + SAFETY_QUERIES[:7] + COMPETITION_QUERIES[:7])
    logger.info(f"Evaluation queries: {len(all_queries)}")

    # ── Extract activations at system_prompt_mean position ─────────────────
    cache_path = OUT / "spm_activations_raw.pt"
    if cache_path.exists():
        logger.info(f"Loading cached activations from {cache_path}")
        activations = ActivationExtractor.load_activations(cache_path)
    else:
        logger.info("Extracting system_prompt_mean activations...")
        activations = extractor.extract_all_conditions(
            queries=all_queries,
            identities=IDENTITIES,
            token_position="system_prompt_mean",
        )
        ActivationExtractor.save_activations(activations, cache_path)

    # Normalize
    norm_activations = ActivationExtractor.normalize_activations(activations)
    logger.info("Activations extracted and normalized")

    # ── Build X, y for probe ───────────────────────────────────────────────
    # Exclude 'none' identity (empty system prompt → undefined system_prompt_mean span).
    # Including it would mix last-token fallback activations with mean-pool activations,
    # polluting the probe dataset (Chen R7).
    IDENTITIES_NO_NONE = {k: v for k, v in IDENTITIES.items() if k != "none"}
    identity_list = list(IDENTITIES_NO_NONE.keys())
    label_map = {k: i for i, k in enumerate(identity_list)}

    X_list = []
    y_list = []
    identity_labels = []

    for identity, query_dict in norm_activations.items():
        if identity not in IDENTITIES_NO_NONE:  # skip 'none' (undefined span)
            continue
        for query, tensor in query_dict.items():
            # tensor: (num_layers, hidden_dim)
            # Peak layer selection: take max across layers by variance
            X_list.append(tensor.numpy())
            y_list.append(label_map[identity])
            identity_labels.append(identity)

    if not X_list:
        logger.error("No activations found — check extraction")
        return

    # Stack: (n_samples, num_layers, hidden_dim)
    X_all = np.array(X_list)
    y_all = np.array(y_list)
    logger.info(f"Data: {X_all.shape[0]} samples, {X_all.shape[1]} layers, {X_all.shape[2]} hidden dim")

    # ── Layer sweep ────────────────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score

    n_layers = X_all.shape[1]
    pca_dim = min(64, X_all.shape[2])

    layer_accs = []
    for layer_idx in range(n_layers):
        X_layer = X_all[:, layer_idx, :]
        pca = PCA(n_components=pca_dim, random_state=42)
        X_pca = pca.fit_transform(X_layer)
        scores = cross_val_score(
            LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=3, max_iter=500),
            X_pca, y_all, cv=3, scoring="accuracy",
        )
        layer_accs.append(float(scores.mean()))

    peak_layer = int(np.argmax(layer_accs))
    peak_acc = layer_accs[peak_layer]
    logger.info(f"\nLayer sweep complete. Peak layer: {peak_layer}, Peak acc: {peak_acc:.4f}")

    # ── Full probe at peak layer ───────────────────────────────────────────
    X_peak = X_all[:, peak_layer, :]
    pca = PCA(n_components=pca_dim, random_state=42)
    X_peak_pca = pca.fit_transform(X_peak)

    X_train, X_test, y_train, y_test = train_test_split(
        X_peak_pca, y_all, test_size=0.2, random_state=42, stratify=y_all,
    )

    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=5, max_iter=500)
    clf.fit(X_train, y_train)
    neural_acc = accuracy_score(y_test, clf.predict(X_test))

    # ── Bag-of-tokens surface baseline ────────────────────────────────────
    from sklearn.feature_extraction.text import CountVectorizer

    # BoW built on system-prompt text ONLY (not full concatenation with query).
    # system_prompt_mean activations see only the system-prompt token span, not
    # the query. Using full concat would give BoW more information than the neural
    # probe, making the comparison unfair (Chen R7 catch).
    texts = []
    y_bow = []
    for identity, sys_prompt in IDENTITIES_NO_NONE.items():
        for _query in all_queries:
            texts.append(sys_prompt)  # system prompt only — matches neural probe's view
            y_bow.append(label_map[identity])
    y_bow = np.array(y_bow)

    vec = CountVectorizer(max_features=5000)
    X_bow = vec.fit_transform(texts).toarray()

    # Trim to same length as neural (some queries may not have activations)
    n_neural = len(X_all)
    X_bow_trim = X_bow[:n_neural]
    y_bow_trim = y_bow[:n_neural]

    X_bow_train, X_bow_test, y_bow_train, y_bow_test = train_test_split(
        X_bow_trim, y_bow_trim, test_size=0.2, random_state=42, stratify=y_bow_trim,
    )
    bow_clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=5, max_iter=500)
    bow_clf.fit(X_bow_train, y_bow_train)
    surface_acc = accuracy_score(y_bow_test, bow_clf.predict(X_bow_test))

    # ── Permutation null ──────────────────────────────────────────────────
    logger.info("Running permutation null (200 reps)...")
    perm_accs = []
    rng = np.random.RandomState(42)
    for _ in range(200):
        y_perm = rng.permutation(y_train)
        clf_perm = LogisticRegressionCV(Cs=[0.1, 1.0], cv=3, max_iter=200)
        clf_perm.fit(X_train, y_perm)
        perm_accs.append(accuracy_score(y_test, clf_perm.predict(X_test)))
    perm_95 = float(np.percentile(perm_accs, 95))

    # ── Verdict ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SYSTEM PROMPT MEAN PROBE RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Peak layer:        {peak_layer}")
    logger.info(f"  Neural probe acc:  {neural_acc:.4f}")
    logger.info(f"  Surface BoW acc:   {surface_acc:.4f}")
    logger.info(f"  Permutation 95th:  {perm_95:.4f}")

    if neural_acc > surface_acc + 0.05:
        verdict = "DISTRIBUTED REPRESENTATION FOUND"
        interp = (
            f"Neural probe ({neural_acc:.2%}) exceeds surface baseline ({surface_acc:.2%}) "
            f"by >{5:.0f}pp. Identity is encoded as a distributed representation "
            f"in the system-prompt token span — not just in lexical surface. "
            f"Phase B probing at `system_prompt_mean` is likely to work."
        )
    elif neural_acc <= perm_95:
        verdict = "NULL — BELOW PERMUTATION NULL"
        interp = (
            f"Neural probe ({neural_acc:.2%}) is at or below the permutation null "
            f"({perm_95:.2%}). Identity does NOT appear as a distributed representation "
            f"even over the system-prompt token span. The Phase A mechanism interpretation "
            f"strengthens: identity operates purely via in-context attention. "
            f"Phase B should NOT expect `system_prompt_mean` probing to work."
        )
    elif neural_acc <= surface_acc + 0.02:
        verdict = "SURFACE ARTIFACT"
        interp = (
            f"Neural probe ({neural_acc:.2%}) ≈ surface baseline ({surface_acc:.2%}). "
            f"Even at system_prompt_mean, probe reads surface token identity, not "
            f"distributed representation. Consistent with Phase A null."
        )
    else:
        verdict = "AMBIGUOUS — MARGINAL SIGNAL"
        interp = (
            f"Neural probe ({neural_acc:.2%}) exceeds permutation null ({perm_95:.2%}) "
            f"but tracks surface baseline ({surface_acc:.2%}). Marginal evidence for "
            f"some distributed encoding. Interpret cautiously."
        )

    logger.info(f"\n  VERDICT: {verdict}")
    logger.info(f"\n  INTERPRETATION:")
    logger.info(f"  {interp}")
    logger.info(f"\n  PHASE B IMPLICATION:")
    if "FOUND" in verdict:
        logger.info("  → Use system_prompt_mean as primary Phase B probe position")
        logger.info("  → Expect above-null probe accuracy in fine-tuned organisms (test at first_response as secondary)")
    else:
        logger.info("  → Use first_response as primary Phase B probe position")
        logger.info("  → Do NOT rely on system_prompt_mean for Phase B (no system prompt in organisms)")
        logger.info("  → A null in fine-tuned organisms would extend the Phase A null: ")
        logger.info("    neither system-prompt conditioning nor fine-tuning creates")
        logger.info("    a detectable distributed identity representation")

    # ── Save results ──────────────────────────────────────────────────────
    import json
    result = {
        "peak_layer": peak_layer,
        "neural_acc": neural_acc,
        "surface_acc": surface_acc,
        "perm_95th": perm_95,
        "verdict": verdict,
        "layer_sweep": layer_accs,
    }
    with open(OUT / "spm_probe_results.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save({"layer_sweep": layer_accs, "peak_layer": peak_layer}, OUT / "spm_layer_sweep.pt")
    logger.info(f"\nResults saved → {OUT}/")
    logger.info(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
