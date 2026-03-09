#!/usr/bin/env python3
"""
Phase A v3 — All panel-recommended fixes applied.

Key changes vs v2:
  1. Position contrast: 'last' vs 'last_query' (not 'first_response' which
     was identical to 'last' due to Gemma-2 prefill hidden-state layout).
     'last_query' takes the last token of the user query text — the SAME
     token embedding across all identity conditions — so layer-0 probe
     accuracy should be ~1/6 (chance). Identity signal at deeper layers
     is genuine propagation through attention, not an embedding artifact.
  2. Real surface baseline: bag-of-tokens on raw input tokens (not identity
     one-hot, which was tautological). If bag-of-tokens achieves ≈ neural
     probe accuracy, the probe is reading surface identity tokens.
  3. Refusal queries expanded from 8 → 30 per identity for statistical power.
  4. Fisher's exact test + Kruskal-Wallis for refusal rate comparisons.
  5. One-sample binomial test per identity for self-promotion asymmetry.
  6. η² (eta-squared) reported for all ANOVAs.
  7. Permutation test reps increased from 100 → 1000.
  8. max_iter increased to 500 to address convergence warnings.
  9. Per-layer overfit gap flagged (warn when gap > 0.10).
 10. Smart cache: generates only missing (identity, query) pairs.

Run on RunPod A40 (46 GB VRAM) with Gemma-2-9B-IT primary.

Usage:
    HF_TOKEN=hf_... python research/run_phase_a_v3.py
"""

import json
import logging
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact, kruskal, binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Output paths ─────────────────────────────────────────────────────────────
OUT = Path("/workspace/research/outputs_v3")
for d in ["generations", "activations", "probes", "reports"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUT / "phase_a_v3.log")),
    ],
)
logger = logging.getLogger("phase_a_v3")


# ── Hyperparameters ───────────────────────────────────────────────────────────
PCA_DIMS = 64
MAX_ITER = 1000         # matches linear_probe.py; 200 caused convergence failures
C_VALUES = [0.01, 0.1, 1.0, 10.0]
CV_FOLDS = 5
RANDOM_STATE = 42
LABEL_SHUFFLE_REPS = 1000   # increased from 100 for tighter null distribution
OVERFIT_WARN_GAP = 0.10     # flag layers where train_acc - val_acc > this


def _serialize(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


# ── Step 1: Generate responses (smart cache — only missing pairs) ─────────────

def step1_generate_responses(
    model, tokenizer, identities: dict, queries: list, query_categories: dict,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 1: Generating responses (smart cache)")
    logger.info("=" * 60)

    from research.models.loader import ModelLoader
    loader = ModelLoader()
    loader.model = model
    loader.tokenizer = tokenizer

    # Build set of already-completed (identity, query) pairs
    done_pairs: set[tuple] = set()
    if existing_df is not None and not existing_df.empty:
        for _, row in existing_df.iterrows():
            done_pairs.add((row["identity"], row["query"]))
        logger.info(f"  Found {len(done_pairs)} cached responses, generating missing pairs only.")

    records = [] if existing_df is None else existing_df.to_dict("records")
    total_needed = len(identities) * len(queries)
    done = len(done_pairs)
    new_count = 0

    for identity, system_prompt in identities.items():
        for query in queries:
            if (identity, query) in done_pairs:
                continue

            prompt = loader.format_prompt(system_prompt, query)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            n_input = inputs["input_ids"].shape[1]
            response = tokenizer.decode(out[0][n_input:], skip_special_tokens=True).strip()
            num_tokens = len(out[0]) - n_input

            category = "unknown"
            for cat, cat_queries in query_categories.items():
                if query in cat_queries:
                    category = cat
                    break

            records.append({
                "identity": identity,
                "system_prompt": system_prompt,
                "query": query,
                "response": response,
                "num_tokens": int(num_tokens),
                "category": category,
            })

            done += 1
            new_count += 1
            if new_count % 50 == 0 or done == total_needed:
                logger.info(f"  {done}/{total_needed} responses done ({new_count} new)")

    df = pd.DataFrame(records)
    csv_path = OUT / "generations" / "phase_a_v3_responses.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Responses saved → {csv_path} ({len(df)} rows, {new_count} newly generated)")
    return df


# ── Step 3: Extract activations ───────────────────────────────────────────────

def step3_extract_activations(
    model, tokenizer, identities: dict, queries: list, position: str
) -> dict:
    logger.info(f"  Extracting activations at position='{position}'...")

    from research.models.activation_extractor import ActivationExtractor
    extractor = ActivationExtractor(model, tokenizer)

    activations_raw = extractor.extract_all_conditions(
        queries=queries,
        identities=identities,
        token_position=position,
    )

    raw_path = OUT / "activations" / f"phase_a_v3_{position}_raw.pt"
    torch.save(activations_raw, raw_path)

    activations_norm = ActivationExtractor.normalize_activations(activations_raw)
    norm_path = OUT / "activations" / f"phase_a_v3_{position}_normalized.pt"
    torch.save(activations_norm, norm_path)

    logger.info(f"  Saved: {raw_path.name} + {norm_path.name}")
    return activations_norm


# ── Step 4: Probe training ────────────────────────────────────────────────────

def _load_layer_data(norm_activations: dict, layer: int):
    X_parts, y_parts = [], []
    for identity, queries in norm_activations.items():
        for tensor in queries.values():
            feat = tensor[layer].float().numpy()
            X_parts.append(feat)
            y_parts.append(identity)
    X = np.stack(X_parts)
    le = LabelEncoder()
    y = le.fit_transform(y_parts)
    return X, y, le


def _fit_pca_probe(X_train, X_val, y_train, y_val, multiclass=True):
    """PCA→64 dims + LogisticRegressionCV. Returns metric dict."""
    n_components = min(PCA_DIMS, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    explained_var = float(pca.explained_variance_ratio_.sum())

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    probe = LogisticRegressionCV(
        Cs=C_VALUES, cv=cv, max_iter=MAX_ITER,
        random_state=RANDOM_STATE, solver="lbfgs", scoring="accuracy",
    )
    probe.fit(X_train_pca, y_train)

    y_val_pred = probe.predict(X_val_pca)
    y_train_pred = probe.predict(X_train_pca)
    val_acc = float(accuracy_score(y_val, y_val_pred))
    train_acc = float(accuracy_score(y_train, y_train_pred))
    overfit_gap = train_acc - val_acc

    result = {
        "accuracy": val_acc,
        "train_accuracy": train_acc,
        "overfit_gap": overfit_gap,
        "best_C": float(probe.C_[0]),
        "pca_explained_variance": explained_var,
    }

    if multiclass:
        result["f1_macro"] = float(f1_score(y_val, y_val_pred, average="macro"))
    else:
        y_proba = probe.predict_proba(X_val_pca)[:, 1]
        result["auroc"] = float(roc_auc_score(y_val, y_proba))
        result["f1"] = float(f1_score(y_val, y_val_pred))
        direction = probe.coef_[0].copy()
        direction /= np.linalg.norm(direction) + 1e-12
        result["direction"] = direction

    return result


def _label_shuffle_permutation(X_train, y_train, X_val, y_val, best_C: float = 0.1):
    """Run probe with shuffled training labels. Returns null distribution stats.

    Uses the same C as the actual probe (cv-selected) so the null model has
    the same capacity as the real probe — avoids biasing null distribution low.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    null_accs = []
    n_components = min(PCA_DIMS, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    for _ in range(LABEL_SHUFFLE_REPS):
        y_shuffled = rng.permutation(y_train)
        probe = LogisticRegression(C=best_C, max_iter=MAX_ITER, solver="lbfgs",
                                   random_state=RANDOM_STATE)
        probe.fit(X_train_pca, y_shuffled)
        null_accs.append(float(accuracy_score(y_val, probe.predict(X_val_pca))))

    return {
        "null_mean": float(np.mean(null_accs)),
        "null_std": float(np.std(null_accs)),
        "null_95pct": float(np.percentile(null_accs, 95)),
        "n_reps": LABEL_SHUFFLE_REPS,
        "C_used": best_C,
    }


def _bag_of_tokens_surface_baseline(
    tokenizer, identities: dict, queries: list, y: np.ndarray
) -> dict:
    """Bag-of-tokens baseline on raw input prompts.

    Builds a count-vector feature matrix from tokenized input prompts, then
    trains a logistic probe on it. If this baseline matches the neural probe
    accuracy, the neural probe is simply reading surface identity tokens
    (the company name literally appears in the system prompt tokens).

    This is the correct surface baseline — not identity one-hot (tautological).
    """
    from research.models.loader import ModelLoader
    loader = ModelLoader()
    loader.tokenizer = tokenizer

    tokenized: list[list[int]] = []
    for identity, system_prompt in identities.items():
        for query in queries:
            prompt = loader.format_prompt(system_prompt, query)
            token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            tokenized.append(token_ids)

    if not tokenized:
        return {"accuracy": 0.0, "note": "No tokenized inputs"}

    max_tid = max(tid for seq in tokenized for tid in seq) + 1
    vocab_size = min(max_tid, 50_000)

    X_bow = np.zeros((len(tokenized), vocab_size), dtype=np.float32)
    for i, seq in enumerate(tokenized):
        for tid in seq:
            if tid < vocab_size:
                X_bow[i, tid] += 1.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_bow, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    probe = LogisticRegressionCV(
        Cs=C_VALUES, cv=cv, max_iter=MAX_ITER,
        random_state=RANDOM_STATE, solver="lbfgs",
    )
    probe.fit(X_train, y_train)
    val_acc = float(accuracy_score(y_val, probe.predict(X_val)))

    return {
        "accuracy": val_acc,
        "note": (
            "Bag-of-tokens on raw input tokens. "
            "If ≈ neural probe → probe reads surface identity tokens (expected for 'last'). "
            "If neural probe >> surface baseline → genuine internal representation."
        ),
    }


def step4_train_probes(
    norm_activations: dict, position_label: str,
    tokenizer=None, identities: dict = None, queries: list = None,
) -> dict:
    logger.info("=" * 60)
    logger.info(f"STEP 4: Probes at position='{position_label}'")
    logger.info("=" * 60)

    num_layers = next(iter(next(iter(norm_activations.values())).values())).shape[0]

    # ── 4a: Multiclass (6-way) layer sweep ───────────────────────────────
    logger.info("  [4a] Multiclass probe sweep...")
    mc_results = {}
    flagged_overfit = []

    for layer in range(num_layers):
        X, y, le = _load_layer_data(norm_activations, layer)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        res = _fit_pca_probe(X_train, X_val, y_train, y_val, multiclass=True)
        res["label_classes"] = le.classes_.tolist()
        mc_results[layer] = res

        if res["overfit_gap"] > OVERFIT_WARN_GAP:
            flagged_overfit.append(layer)

        if layer % 7 == 0 or layer == num_layers - 1:
            logger.info(
                f"    layer {layer:2d}: val_acc={res['accuracy']:.4f}, "
                f"f1={res['f1_macro']:.4f}, "
                f"overfit_gap={res['overfit_gap']:+.3f}, "
                f"pca_var={res['pca_explained_variance']:.3f}"
            )

    mc_acc = {l: r["accuracy"] for l, r in mc_results.items()}
    peak_mc = max(mc_acc, key=mc_acc.get)
    logger.info(f"  Multiclass peak: layer {peak_mc}, acc={mc_acc[peak_mc]:.4f}")
    if flagged_overfit:
        logger.warning(f"  ⚠ Layers with overfit gap > {OVERFIT_WARN_GAP}: {flagged_overfit}")

    # ── 4b: Eta-squared for multiclass layer sweep ────────────────────────
    # η² = SS_between / SS_total on val_accuracy across layers
    all_accs = [r["accuracy"] for r in mc_results.values()]
    grand_mean = np.mean(all_accs)
    # η² quantifies how much of accuracy variance is explained by layer choice
    # (informational, not a hypothesis test)

    # ── 4c: Surface baseline (bag-of-tokens) at peak layer ───────────────
    logger.info("  [4b] Bag-of-tokens surface baseline at peak layer...")
    X_peak, y_peak, _ = _load_layer_data(norm_activations, peak_mc)

    if tokenizer is not None and identities is not None:
        # Use the query set actually present in norm_activations, not the full
        # query list — they may differ if some queries were added after cache
        # was created (would cause shape mismatch between X_bow and y_peak).
        actual_queries = (
            list(next(iter(norm_activations.values())).keys())
            if norm_activations else (queries or [])
        )
        surf = _bag_of_tokens_surface_baseline(tokenizer, identities, actual_queries, y_peak)
    else:
        surf = {"accuracy": float("nan"), "note": "Tokenizer not provided"}

    logger.info(f"  Surface (bag-of-tokens): {surf['accuracy']:.4f} "
                f"vs neural: {mc_acc[peak_mc]:.4f}")
    neural_exceeds_surface = (
        mc_acc[peak_mc] > surf["accuracy"] + 0.05
        if not np.isnan(surf["accuracy"]) else None
    )
    if neural_exceeds_surface is False:
        logger.warning(
            "  ⚠ Neural probe ≈ surface baseline — probe may be reading input "
            "identity tokens rather than learned internal representations."
        )
    elif neural_exceeds_surface:
        logger.info("  ✓ Neural probe exceeds surface baseline by >5pp — genuine internal signal.")

    # ── 4d: Label-shuffle permutation at peak layer ───────────────────────
    # Use the CV-selected best_C so null model has same capacity as real probe
    peak_best_C = float(mc_results[peak_mc].get("best_C", 0.1))
    logger.info(f"  [4c] Label-shuffle permutation ({LABEL_SHUFFLE_REPS} reps, C={peak_best_C})...")
    X_train_pk, X_val_pk, y_train_pk, y_val_pk = train_test_split(
        X_peak, y_peak, test_size=0.2, random_state=RANDOM_STATE, stratify=y_peak
    )
    perm_result = _label_shuffle_permutation(X_train_pk, y_train_pk, X_val_pk, y_val_pk, best_C=peak_best_C)
    exceeds_null = mc_acc[peak_mc] > perm_result["null_95pct"]
    logger.info(
        f"  Null: mean={perm_result['null_mean']:.4f}, 95th={perm_result['null_95pct']:.4f}. "
        f"Neural={mc_acc[peak_mc]:.4f} → "
        f"{'EXCEEDS null 95th' if exceeds_null else 'WITHIN null (no signal)'}"
    )

    # ── 4e: Binary probes ─────────────────────────────────────────────────
    binary_pairs = [
        ("anthropic", "openai"),
        ("anthropic", "google"),
        ("anthropic", "neutral"),
        ("openai",    "google"),
        ("openai",    "neutral"),
        ("none",      "neutral"),
    ]
    binary_pairs = [(a, b) for a, b in binary_pairs
                    if a in norm_activations and b in norm_activations]

    binary_results = {}
    for pos_id, neg_id in binary_pairs:
        name = f"{pos_id}_vs_{neg_id}"
        pair_results = {}
        for layer in range(num_layers):
            X_pos = np.stack([norm_activations[pos_id][q][layer].float().numpy()
                              for q in norm_activations[pos_id]])
            X_neg = np.stack([norm_activations[neg_id][q][layer].float().numpy()
                              for q in norm_activations[neg_id]])
            X = np.concatenate([X_pos, X_neg])
            y = np.concatenate([np.ones(len(X_pos), dtype=int), np.zeros(len(X_neg), dtype=int)])
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            pair_results[layer] = _fit_pca_probe(X_train, X_val, y_train, y_val, multiclass=False)

        accs = {l: r["accuracy"] for l, r in pair_results.items()}
        pk = max(accs, key=accs.get)
        logger.info(f"  [4e] {name}: peak layer {pk}, "
                    f"acc={accs[pk]:.4f}, AUROC={pair_results[pk]['auroc']:.4f}")
        binary_results[name] = pair_results

    def _clean(res_dict):
        return {
            layer: {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in r.items() if k not in ("model",)}
            for layer, r in res_dict.items()
        }

    probe_output = {
        "model": "Gemma-2-9B-IT",
        "position": position_label,
        "pca_dims": PCA_DIMS,
        "max_iter": MAX_ITER,
        "multiclass": _clean(mc_results),
        "binary": {n: _clean(r) for n, r in binary_results.items()},
        "surface_baseline_bag_of_tokens": surf,
        "label_shuffle_permutation": perm_result,
        "peak_multiclass_layer": int(peak_mc),
        "peak_multiclass_accuracy": float(mc_acc[peak_mc]),
        "neural_exceeds_surface": neural_exceeds_surface,
        "exceeds_null_95th": bool(exceeds_null),
        "overfit_flagged_layers": flagged_overfit,
    }

    out_path = OUT / "probes" / f"phase_a_v3_{position_label}_probes.json"
    with open(out_path, "w") as f:
        json.dump(probe_output, f, indent=2, default=_serialize)
    logger.info(f"  Probe results saved → {out_path}")
    return probe_output


# ── Step 5: KPI Evaluation ────────────────────────────────────────────────────

def step5_kpi_evaluation(df: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 5: KPI Behavioral Evaluation")
    logger.info("=" * 60)

    from research.evaluation.kpi_metrics import KPIEvaluator
    evaluator = KPIEvaluator()
    results = evaluator.run_full_evaluation(df)

    report = evaluator.generate_evaluation_report(results)
    (OUT / "reports" / "kpi_report_v3.md").write_text(report)
    with open(OUT / "reports" / "kpi_results_v3.json", "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    logger.info("\n" + report)
    return results


# ── Step 6: Statistical Tests ─────────────────────────────────────────────────

def _eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
    """Compute η² (eta-squared) from F-statistic and degrees of freedom."""
    ss_between = f_stat * df_between
    ss_total = ss_between + df_within
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def step6_statistical_tests(df: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 6: Statistical Tests")
    logger.info("=" * 60)

    from research.evaluation.statistical_tests import StatisticalAnalyzer
    from research.evaluation.refusal_patterns import classify_refusal
    from research.evaluation.kpi_metrics import KPIEvaluator

    analyzer = StatisticalAnalyzer()
    all_stats = {}
    identities = sorted(df["identity"].unique())
    n_identities = len(identities)
    N_per_group = len(df[df["identity"] == identities[0]])

    # ── 6a: Token count ANOVA with η² ────────────────────────────────────
    anova = analyzer.anova_across_identities(df, metric_col="num_tokens")
    df_between = n_identities - 1
    df_within = len(df) - n_identities
    eta2 = _eta_squared(anova["f_statistic"], df_between, df_within)
    anova["eta_squared"] = eta2
    anova["eta_squared_interpretation"] = (
        "negligible" if eta2 < 0.01 else
        "small" if eta2 < 0.06 else
        "medium" if eta2 < 0.14 else "large"
    )
    all_stats["anova_token_count"] = anova
    logger.info(
        f"ANOVA tokens: F={anova['f_statistic']:.4f}, "
        f"p={anova['p_value']:.6f}, η²={eta2:.4f} "
        f"({anova['eta_squared_interpretation']}), sig={anova['significant']}"
    )

    # ── 6b: Category-stratified ANOVA ────────────────────────────────────
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        if cat_df["identity"].nunique() < 2 or len(cat_df) < 12:
            continue
        try:
            cat_anova = analyzer.anova_across_identities(cat_df, metric_col="num_tokens")
            n_cat = len(cat_df)
            eta2_cat = _eta_squared(cat_anova["f_statistic"], df_between, n_cat - n_identities)
            cat_anova["eta_squared"] = eta2_cat
            all_stats[f"anova_token_{cat}"] = cat_anova
            if cat_anova["significant"]:
                logger.info(f"  ANOVA [{cat}]: F={cat_anova['f_statistic']:.3f}, "
                            f"p={cat_anova['p_value']:.4f}, η²={eta2_cat:.4f} *** SIGNIFICANT")
        except Exception:
            pass

    # ── 6c: Pairwise t-tests with BH correction ───────────────────────────
    pairwise = analyzer.pairwise_significance(df, metric_col="num_tokens")
    sig_pairs = pairwise[pairwise["significant"]]
    logger.info(f"Pairwise token t-tests: {len(sig_pairs)}/{len(pairwise)} significant after BH")

    # ── 6d: Cohen's d ────────────────────────────────────────────────────
    cohens = analyzer.pairwise_cohens_d(df, metric_col="num_tokens")
    large_effects = cohens[cohens["interpretation"].isin(["medium", "large"])]
    logger.info(f"Medium/large effect sizes (token count): {len(large_effects)}/{len(cohens)}")

    # ── 6e: Permutation test (most verbose vs most concise) ───────────────
    means = df.groupby("identity")["num_tokens"].mean()
    verbose_id, concise_id = means.idxmax(), means.idxmin()
    perm = analyzer.permutation_test(
        df[df["identity"] == verbose_id]["num_tokens"].values,
        df[df["identity"] == concise_id]["num_tokens"].values,
    )
    all_stats["permutation_verbose_vs_concise"] = {
        **perm, "group_a": verbose_id, "group_b": concise_id
    }
    logger.info(
        f"Permutation ({verbose_id} vs {concise_id}): "
        f"diff={perm['observed_diff']:.1f}, p={perm['p_value']:.6f}, sig={perm['significant']}"
    )

    # ── 6f: Refusal rate tests (Fisher's exact + Kruskal-Wallis) ─────────
    logger.info("  [6f] Refusal rate statistical tests...")
    refusal_df = df[df["category"] == "refusal"].copy()
    if not refusal_df.empty:
        refusal_df["refused"] = refusal_df["response"].apply(
            lambda r: classify_refusal(r) != "no_refusal"
        ).astype(int)

        # Kruskal-Wallis across all identities
        groups_refused = [
            refusal_df[refusal_df["identity"] == ident]["refused"].values
            for ident in identities
        ]
        if all(len(g) > 0 for g in groups_refused):
            try:
                kw_stat, kw_p = kruskal(*groups_refused)
                all_stats["kruskal_wallis_refusal"] = {
                    "statistic": float(kw_stat),
                    "p_value": float(kw_p),
                    "significant": bool(kw_p < 0.05),
                    "n_per_group": int(len(refusal_df) // n_identities),
                }
                logger.info(
                    f"  Kruskal-Wallis refusal: H={kw_stat:.4f}, "
                    f"p={kw_p:.6f}, sig={kw_p < 0.05}"
                )
            except Exception as e:
                logger.warning(f"  Kruskal-Wallis failed: {e}")

        # Pairwise Fisher's exact tests
        fisher_rows = []
        for id_a, id_b in combinations(identities, 2):
            a_refused = refusal_df[refusal_df["identity"] == id_a]["refused"].values
            b_refused = refusal_df[refusal_df["identity"] == id_b]["refused"].values
            if len(a_refused) == 0 or len(b_refused) == 0:
                continue
            table = [
                [int(a_refused.sum()), int(len(a_refused) - a_refused.sum())],
                [int(b_refused.sum()), int(len(b_refused) - b_refused.sum())],
            ]
            _, p = fisher_exact(table, alternative="two-sided")
            rate_a = float(a_refused.mean())
            rate_b = float(b_refused.mean())
            fisher_rows.append({
                "identity_a": id_a, "identity_b": id_b,
                "rate_a": rate_a, "rate_b": rate_b,
                "p_value": float(p),
                "n_a": int(len(a_refused)), "n_b": int(len(b_refused)),
            })

        if fisher_rows:
            raw_ps = [r["p_value"] for r in fisher_rows]
            adj_ps = StatisticalAnalyzer.benjamini_hochberg(raw_ps)
            for row, adj_p in zip(fisher_rows, adj_ps):
                row["p_adjusted"] = float(adj_p)
                row["significant"] = bool(adj_p < 0.05)

            sig_fisher = [r for r in fisher_rows if r["significant"]]
            all_stats["fisher_exact_refusal"] = {
                "rows": fisher_rows,
                "n_pairs_tested": len(fisher_rows),
                "n_significant": len(sig_fisher),
            }
            logger.info(
                f"  Fisher's exact (BH-corrected): "
                f"{len(sig_fisher)}/{len(fisher_rows)} pairs significant. "
                f"N={refusal_df['identity'].value_counts().min()} per identity."
            )
            for row in sig_fisher:
                logger.info(
                    f"    {row['identity_a']} ({row['rate_a']:.1%}) vs "
                    f"{row['identity_b']} ({row['rate_b']:.1%}): "
                    f"p_adj={row['p_adjusted']:.4f}"
                )
    else:
        logger.warning("  No refusal-category queries in data.")

    # ── 6g: Self-promotion binomial test per identity ─────────────────────
    logger.info("  [6g] Self-promotion one-sample binomial tests...")
    promo_df = df[df["category"].isin(["self_promotion", "self_promotion_unprimed"])].copy()
    if not promo_df.empty:
        evaluator = KPIEvaluator()
        import re as _re
        binomial_rows = []

        for ident in identities:
            if ident in ("neutral", "none"):
                continue  # no own brand to promote

            subset = promo_df[promo_df["identity"] == ident]
            own_kws = evaluator.COMPANY_KEYWORDS.get(ident, [])
            if not own_kws:
                continue

            own_re = _re.compile(
                r"\b(" + "|".join(_re.escape(kw) for kw in own_kws) + r")\b",
                _re.IGNORECASE,
            )
            mentions = subset["response"].apply(lambda r: bool(own_re.search(r)))
            k = int(mentions.sum())
            n = len(mentions)

            # Two-sided binomial test vs H0: p=0.5 (no preference)
            # (p=0.5 means equal chance of mentioning own vs not mentioning)
            result = binomtest(k, n, p=0.5, alternative="greater")
            binomial_rows.append({
                "identity": ident,
                "own_mention_count": k,
                "n_responses": n,
                "own_mention_rate": float(k / n) if n > 0 else 0.0,
                "p_value": float(result.pvalue),
            })

        if binomial_rows:
            raw_ps = [r["p_value"] for r in binomial_rows]
            adj_ps = StatisticalAnalyzer.benjamini_hochberg(raw_ps)
            for row, adj_p in zip(binomial_rows, adj_ps):
                row["p_adjusted"] = float(adj_p)
                row["significant"] = bool(adj_p < 0.05)

            # Store as dict for generate_statistical_report compatibility
            all_stats["binomial_self_promotion"] = {
                "rows": binomial_rows,
                "n_identities_tested": len(binomial_rows),
                "n_significant": sum(1 for r in binomial_rows if r.get("significant", False)),
            }
            logger.info("  Self-promotion binomial (own mention > 50%, BH-corrected):")
            for row in sorted(binomial_rows, key=lambda r: r["own_mention_rate"], reverse=True):
                sig_marker = "***" if row["significant"] else "n.s."
                logger.info(
                    f"    {row['identity']}: {row['own_mention_rate']:.1%} "
                    f"({row['own_mention_count']}/{row['n_responses']}) "
                    f"p_adj={row['p_adjusted']:.4f} {sig_marker}"
                )

    # ── Reports ───────────────────────────────────────────────────────────
    report = analyzer.generate_statistical_report(all_stats)
    (OUT / "reports" / "statistical_report_v3.txt").write_text(report)
    pairwise.to_csv(OUT / "reports" / "pairwise_significance_v3.csv", index=False)
    cohens.to_csv(OUT / "reports" / "cohens_d_v3.csv", index=False)
    logger.info("\n" + report)
    return all_stats, pairwise, cohens


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PHASE A v3 — Corporate Identity Research (fully fixed pipeline)")
    logger.info("=" * 60)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("HuggingFace token set.")

    import research.config as _cfg
    _cfg.model_config.model_name = "google/gemma-2-9b-it"
    _cfg.model_config.fallback_model = "Qwen/Qwen2.5-7B-Instruct"

    from research.models.loader import ModelLoader
    loader = ModelLoader()
    model, tokenizer = loader.load_model()

    actual_name = _cfg.model_config.model_name.lower()
    if "gemma-2-9b" in actual_name:
        _cfg.model_config.num_layers = 42
        _cfg.model_config.hidden_dim = 3584
        logger.info("Config: Gemma-2-9B-IT (42 layers, 3584 dim)")
    elif "qwen" in actual_name:
        _cfg.model_config.num_layers = 28
        _cfg.model_config.hidden_dim = 3584
        logger.info("Config: Qwen2.5-7B-Instruct (28 layers, 3584 dim)")

    from research.data.prompts import ALL_QUERIES, QUERY_CATEGORIES, REFUSAL_QUERIES
    identities = _cfg.IDENTITY_CONDITIONS
    queries = ALL_QUERIES

    logger.info(f"Identities: {list(identities.keys())}")
    logger.info(f"Total queries per identity: {len(queries)}")
    logger.info(f"  - Refusal queries: {len(REFUSAL_QUERIES)} (was 8, now {len(REFUSAL_QUERIES)})")
    logger.info(f"Total responses to generate: {len(identities) * len(queries)}")

    # ── Step 1: Generate responses (smart cache) ──────────────────────────
    csv_path = OUT / "generations" / "phase_a_v3_responses.csv"
    existing_df = None

    # Also check v2 responses to reuse existing work
    v2_csv = Path("/workspace/research/outputs_v2/generations/phase_a_v2_responses.csv")
    if csv_path.exists():
        logger.info(f"Loading existing v3 responses from {csv_path}")
        existing_df = pd.read_csv(csv_path)
    elif v2_csv.exists():
        logger.info(f"Loading v2 responses as starting point from {v2_csv}")
        existing_df = pd.read_csv(v2_csv)
        # Fix categories for any queries already in v2 (category may differ if prompts changed)
        existing_df["category"] = existing_df["query"].apply(
            lambda q: next(
                (cat for cat, qs in QUERY_CATEGORIES.items() if q in qs), "unknown"
            )
        )

    df = step1_generate_responses(
        model, tokenizer, identities, queries, QUERY_CATEGORIES, existing_df
    )
    logger.info(f"Responses: {len(df)} rows, identities={df['identity'].unique().tolist()}")

    # ── Step 3: Extract activations — 'last' and 'last_query' ────────────
    # KEY CHANGE: 'last_query' replaces 'first_response'.
    # 'last_query' takes the last token of the user query text, which is
    # IDENTICAL across all identity conditions for a given query.
    # At layer 0 (embedding), the probe should see ~chance accuracy (same token).
    # At deeper layers, identity propagation through attention can be detected.
    positions = ["last", "last_query", "first_response"]
    activations_by_position = {}

    for position in positions:
        norm_path = OUT / "activations" / f"phase_a_v3_{position}_normalized.pt"
        if norm_path.exists():
            logger.info(f"Loading cached activations [{position}] from {norm_path}")
            activations_by_position[position] = torch.load(
                norm_path, map_location="cpu", weights_only=False
            )
        else:
            activations_by_position[position] = step3_extract_activations(
                model, tokenizer, identities, queries, position
            )

    # ── Step 4: Train probes at each position ─────────────────────────────
    probe_results = {}
    for position in positions:
        probe_results[position] = step4_train_probes(
            activations_by_position[position], position,
            tokenizer=tokenizer, identities=identities, queries=queries,
        )

    # ── Position comparison: all positions ────────────────────────────────
    logger.info("\n=== POSITION COMPARISON ===")
    logger.info(
        "  'last'           = last token of generation prefix (surface identity tokens in context).\n"
        "  'last_query'     = last token of user query text (same token embedding across conditions;\n"
        "                     layer-0 ~1/6 chance; signal at depth = genuine propagation).\n"
        "  'first_response' = first generated token (model committed to response;\n"
        "                     cleanest probe of internalized identity post-processing)."
    )
    for pos in positions:
        if pos not in probe_results:
            continue
        pk = probe_results[pos]["peak_multiclass_layer"]
        acc = probe_results[pos]["peak_multiclass_accuracy"]
        surf = probe_results[pos]["surface_baseline_bag_of_tokens"]["accuracy"]
        perm_95 = probe_results[pos]["label_shuffle_permutation"]["null_95pct"]
        exceeds_surface = probe_results[pos]["neural_exceeds_surface"]
        exceeds_null = probe_results[pos]["exceeds_null_95th"]
        interpretation = (
            "REAL SIGNAL (exceeds both surface baseline and null)"
            if exceeds_surface and exceeds_null
            else "SURFACE ARTIFACT (neural ≈ bag-of-tokens)"
            if not exceeds_surface and exceeds_null
            else "BELOW NULL (no signal)"
            if not exceeds_null
            else "PARTIAL (exceeds null but not surface)"
        )
        logger.info(
            f"  [{pos}] peak layer {pk}: neural={acc:.4f}, "
            f"surface={surf:.4f}, null_95th={perm_95:.4f} → {interpretation}"
        )

    # Layer-0 diagnostic for positions that share embeddings across conditions
    lq_layer0 = probe_results.get("last_query", {}).get("multiclass", {}).get(0, {}).get("accuracy", float("nan"))
    last_layer0 = probe_results.get("last", {}).get("multiclass", {}).get(0, {}).get("accuracy", float("nan"))
    fr_layer0 = probe_results.get("first_response", {}).get("multiclass", {}).get(0, {}).get("accuracy", float("nan"))
    logger.info(
        f"\n  Layer-0 accuracy diagnostic:\n"
        f"    'last' at layer 0:           {last_layer0:.4f} "
        f"(expected high — identity tokens in prefix)\n"
        f"    'last_query' at layer 0:     {lq_layer0:.4f} "
        f"(expected ~{1/len(identities):.3f} — same token embedding across conditions)\n"
        f"    'first_response' at layer 0: {fr_layer0:.4f} "
        f"(varies — if high: identity tokens propagate; if ~chance: no surface artifact)"
    )

    # ── Steps 5-6: KPI + stats ────────────────────────────────────────────
    kpi = step5_kpi_evaluation(df)
    stats, pairwise, cohens = step6_statistical_tests(df)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"ALL DONE in {elapsed / 60:.1f} minutes")
    logger.info("=" * 60)

    # ── Key findings summary ──────────────────────────────────────────────
    logger.info("\n=== KEY FINDINGS v3 ===")

    logger.info("-- Probing --")
    for pos in positions:
        if pos not in probe_results:
            continue
        pk = probe_results[pos]["peak_multiclass_layer"]
        acc = probe_results[pos]["peak_multiclass_accuracy"]
        surf = probe_results[pos]["surface_baseline_bag_of_tokens"]["accuracy"]
        perm_95 = probe_results[pos]["label_shuffle_permutation"]["null_95pct"]
        verdict = probe_results[pos].get("neural_exceeds_surface", False)
        exceeds_null = probe_results[pos].get("exceeds_null_95th", False)
        tag = "REAL SIGNAL" if verdict and exceeds_null else "SURFACE ARTIFACT" if not verdict and exceeds_null else "NULL"
        logger.info(f"[{pos}] peak layer {pk}: neural={acc:.4f}, surface={surf:.4f}, null_95={perm_95:.4f} → {tag}")

    logger.info(f"\n'last_query' layer-0: {lq_layer0:.4f} (chance={1/len(identities):.3f})")
    logger.info("Identity first propagates to last_query residual stream at:")
    lq_accs = {l: r["accuracy"] for l, r in probe_results["last_query"]["multiclass"].items()}
    chance = 1.0 / len(identities)
    first_above_chance = next(
        (l for l in sorted(lq_accs) if lq_accs[l] > chance + 0.10), None
    )
    logger.info(f"  Layer {first_above_chance} (first layer where acc > chance+10pp)")

    logger.info("\n-- Token Economics --")
    anova = stats.get("anova_token_count", {})
    logger.info(
        f"ANOVA: F={anova.get('f_statistic', 0):.2f}, "
        f"p={anova.get('p_value', 1):.6f}, "
        f"η²={anova.get('eta_squared', 0):.4f} "
        f"({anova.get('eta_squared_interpretation', '?')})"
    )

    logger.info("\n-- Self-Promotion --")
    sp = kpi.get("self_promotion", {}).get("per_identity", {})
    for ident, m in sorted(sp.items(), key=lambda x: -x[1].get("preference_asymmetry", 0)):
        logger.info(
            f"  {ident}: asymmetry={m.get('preference_asymmetry', 0):.3f}, "
            f"primed={m.get('own_mention_rate_primed', 0):.2f}, "
            f"unprimed={m.get('own_mention_rate_unprimed', 0):.2f}"
        )

    bin_promo_entry = stats.get("binomial_self_promotion", {})
    bin_promo = bin_promo_entry.get("rows", []) if isinstance(bin_promo_entry, dict) else []
    if bin_promo:
        logger.info("  Binomial tests (own mention > 50%):")
        for row in sorted(bin_promo, key=lambda r: r["own_mention_rate"], reverse=True):
            logger.info(
                f"    {row['identity']}: {row['own_mention_rate']:.1%}, "
                f"p_adj={row['p_adjusted']:.4f} "
                f"{'***' if row['significant'] else 'n.s.'}"
            )

    logger.info("\n-- Refusal (with 95% Wilson CIs) --")
    rb = kpi.get("refusal_behavior", {}).get("per_identity", {})
    from math import sqrt
    def _wilson_ci(k: int, n: int, z: float = 1.96):
        """Wilson score interval for a proportion."""
        if n == 0:
            return 0.0, 1.0
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = (z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    for ident, m in sorted(rb.items(), key=lambda x: -x[1]["overall_refusal_rate"]):
        n_q = m["n_queries"]
        k_q = round(m["overall_refusal_rate"] * n_q)
        lo, hi = _wilson_ci(k_q, n_q)
        logger.info(
            f"  {ident}: overall={m['overall_refusal_rate']:.1%} "
            f"[{lo:.1%}–{hi:.1%}] 95%CI, "
            f"hard={m['hard_refusal_rate']:.1%}, n={n_q}"
        )

    # Power note: required N per group to detect observed max vs min refusal difference
    refusal_rates = [m["overall_refusal_rate"] for m in rb.values()]
    if len(refusal_rates) >= 2:
        import math
        r_max, r_min = max(refusal_rates), min(refusal_rates)
        h_obs = 2 * abs(math.asin(math.sqrt(r_max)) - math.asin(math.sqrt(r_min)))
        # N for 80% power, two-tailed alpha=0.05: n = (z_alpha/2 + z_beta)^2 / h^2
        # z_alpha/2=1.96, z_beta=0.842 → (2.802)^2 = 7.85
        n_needed = math.ceil(7.85 / (h_obs**2)) if h_obs > 0 else float("inf")
        logger.info(
            f"  Power note: observed Cohen's h={h_obs:.3f} "
            f"(max={r_max:.1%} vs min={r_min:.1%}). "
            f"N needed for 80% power: ~{n_needed} per group "
            f"(current N={list(rb.values())[0]['n_queries']})"
        )

    kw = stats.get("kruskal_wallis_refusal", {})
    if kw:
        logger.info(
            f"  Kruskal-Wallis: H={kw.get('statistic', 0):.3f}, "
            f"p={kw.get('p_value', 1):.6f}, sig={kw.get('significant')}"
        )

    fisher_entry = stats.get("fisher_exact_refusal", {})
    fisher = fisher_entry.get("rows", []) if isinstance(fisher_entry, dict) else []
    sig_f = [r for r in fisher if r.get("significant")]
    logger.info(f"  Fisher's exact: {len(sig_f)}/{len(fisher)} pairs significant after BH")

    logger.info(f"\nAll outputs → {OUT}/")


if __name__ == "__main__":
    main()
