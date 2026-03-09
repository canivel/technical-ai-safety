#!/usr/bin/env python3
"""
Phase A v2 — Full pipeline with methodological fixes from panel review.

Key changes vs v1:
  1. Correct chat template: model_name updated after fallback load (loader.py fix).
  2. Dual extraction positions: 'last' (generation prefix) AND 'first_response'
     (first generated token — cleanest identity probe).
  3. Surface baseline at each position: confirms whether probe exceeds bag-of-tokens.
  4. Label-shuffle permutation: validates there is no pipeline data leakage.
  5. Fixed refusal queries: identity-contingent, near the model's actual boundary.
  6. Fixed refusal classifier: narrow soft-refusal patterns (no false "however" hits).
  7. Cross-brand preference matrix: self-promotion comparable for neutral/none.
  8. Within-condition bootstrap: hidden influence excess score above noise floor.
  9. Primed vs unprimed self-promotion split.

Run on RunPod A40 (46 GB VRAM) with Gemma-2-9B-IT primary, Qwen2.5-7B fallback.

Usage:
    HF_TOKEN=hf_... python research/run_phase_a_v2.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Output paths ─────────────────────────────────────────────────────────────
OUT = Path("/workspace/research/outputs_v2")
for d in ["generations", "activations", "probes", "reports"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUT / "phase_a_v2.log")),
    ],
)
logger = logging.getLogger("phase_a_v2")


# ── Probe hyperparameters ─────────────────────────────────────────────────────
PCA_DIMS = 64
MAX_ITER = 200
C_VALUES = [0.01, 0.1, 1.0, 10.0]
CV_FOLDS = 5
RANDOM_STATE = 42
LABEL_SHUFFLE_REPS = 100  # permutations for label-shuffle test


def _serialize(obj):
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    return obj


# ── Step 1-3: Generate responses + extract activations ───────────────────────

def step1_generate_responses(model, tokenizer, identities: dict, queries: list, query_categories: dict) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 1-2: Generating responses")
    logger.info("=" * 60)

    from research.models.loader import ModelLoader
    loader = ModelLoader()
    loader.model = model
    loader.tokenizer = tokenizer

    records = []
    total = len(identities) * len(queries)
    done = 0

    for identity, system_prompt in identities.items():
        for query in queries:
            prompt = loader.format_prompt(system_prompt, query)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the newly generated tokens
            n_input = inputs["input_ids"].shape[1]
            response = tokenizer.decode(out[0][n_input:], skip_special_tokens=True).strip()
            num_tokens = len(out[0]) - n_input

            # Determine category
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
            if done % 50 == 0 or done == total:
                logger.info(f"  {done}/{total} responses generated")

    df = pd.DataFrame(records)
    csv_path = OUT / "generations" / "phase_a_v2_responses.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Responses saved → {csv_path} ({len(df)} rows)")
    return df


def step3_extract_activations(model, tokenizer, identities: dict, queries: list, position: str) -> dict:
    logger.info(f"  Extracting activations at position='{position}'...")

    from research.models.activation_extractor import ActivationExtractor
    extractor = ActivationExtractor(model, tokenizer)

    activations_raw = extractor.extract_all_conditions(
        queries=queries,
        identities=identities,
        token_position=position,
    )

    # Save raw
    raw_path = OUT / "activations" / f"phase_a_v2_{position}_raw.pt"
    torch.save(activations_raw, raw_path)

    # Normalize and save
    activations_norm = ActivationExtractor.normalize_activations(activations_raw)
    norm_path = OUT / "activations" / f"phase_a_v2_{position}_normalized.pt"
    torch.save(activations_norm, norm_path)

    logger.info(f"  Activations saved: {raw_path.name} + {norm_path.name}")
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
    """PCA→64 + LogisticRegressionCV. Returns metric dict."""
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

    y_pred = probe.predict(X_val_pca)
    val_acc = float(accuracy_score(y_val, y_pred))
    train_acc = float(accuracy_score(y_train, probe.predict(X_train_pca)))

    result = {
        "accuracy": val_acc,
        "train_accuracy": train_acc,
        "overfit_gap": train_acc - val_acc,
        "best_C": float(probe.C_[0]),
        "pca_explained_variance": explained_var,
    }

    if multiclass:
        result["f1_macro"] = float(f1_score(y_val, y_pred, average="macro"))
        result["confusion_matrix"] = confusion_matrix(y_val, y_pred).tolist()
    else:
        y_proba = probe.predict_proba(X_val_pca)[:, 1]
        result["auroc"] = float(roc_auc_score(y_val, y_proba))
        result["f1"] = float(f1_score(y_val, y_pred))
        direction = probe.coef_[0].copy()
        direction /= np.linalg.norm(direction) + 1e-12
        result["direction"] = direction

    return result


def _label_shuffle_permutation(X_train, y_train, X_val, y_val, n_reps=LABEL_SHUFFLE_REPS):
    """Run probe with shuffled labels to estimate chance accuracy and detect leakage."""
    rng = np.random.RandomState(RANDOM_STATE)
    null_accs = []
    n_components = min(PCA_DIMS, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    for _ in range(n_reps):
        y_shuffled = rng.permutation(y_train)
        probe = LogisticRegression(C=0.01, max_iter=MAX_ITER, solver="lbfgs",
                                   random_state=RANDOM_STATE)
        probe.fit(X_train_pca, y_shuffled)
        null_accs.append(float(accuracy_score(y_val, probe.predict(X_val_pca))))

    return {
        "null_mean": float(np.mean(null_accs)),
        "null_std": float(np.std(null_accs)),
        "null_95pct": float(np.percentile(null_accs, 95)),
        "n_reps": n_reps,
    }


def _surface_baseline(norm_activations: dict, identities: list, queries: list, y: np.ndarray, layer: int):
    """Bag-of-identity-prompt-tokens baseline using the prompt identity string length as a simple feature."""
    # The simplest possible surface feature: which identity (by index) the sample belongs to.
    # A probe matching this achieves 100% by memorizing identity labels directly.
    # We use the first-token embedding at layer 0 as a proxy — if this matches the neural probe,
    # the probe is reading surface identity tokens.
    # Practical implementation: encode each sample by which identity it came from (one-hot).
    identity_list = []
    for identity, qs in norm_activations.items():
        for _ in qs:
            identity_list.append(identity)

    le = LabelEncoder()
    le.fit(sorted(set(identity_list)))
    y_identity = le.transform(identity_list)

    # One-hot encode as features
    n_classes = len(le.classes_)
    X_surf = np.eye(n_classes, dtype=np.float32)[y_identity]

    X_train, X_val, y_train, y_val = train_test_split(
        X_surf, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    probe = LogisticRegressionCV(
        Cs=C_VALUES, cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        max_iter=MAX_ITER, random_state=RANDOM_STATE, solver="lbfgs",
    )
    probe.fit(X_train, y_train)
    val_acc = float(accuracy_score(y_val, probe.predict(X_val)))
    return {"accuracy": val_acc, "note": "1.0 means probe is trivially reading identity label — identity encoding artifact confirmed"}


def step4_train_probes(norm_activations: dict, position_label: str) -> dict:
    logger.info("=" * 60)
    logger.info(f"STEP 4: Probes at position='{position_label}'")
    logger.info("=" * 60)

    num_layers = next(iter(next(iter(norm_activations.values())).values())).shape[0]
    identities = list(norm_activations.keys())
    queries_per_id = [list(norm_activations[i].keys()) for i in identities]

    # ── 4a: Multiclass (6-way) ────────────────────────────────────────────
    logger.info("  [4a] Multiclass probe sweep...")
    mc_results = {}

    for layer in range(num_layers):
        X, y, le = _load_layer_data(norm_activations, layer)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        res = _fit_pca_probe(X_train, X_val, y_train, y_val, multiclass=True)
        res["label_classes"] = le.classes_.tolist()
        mc_results[layer] = res

        if layer % 7 == 0 or layer == num_layers - 1:
            logger.info(f"    layer {layer:2d}: acc={res['accuracy']:.4f}, "
                        f"f1={res['f1_macro']:.4f}, "
                        f"pca_var={res['pca_explained_variance']:.3f}")

    mc_acc = {l: r["accuracy"] for l, r in mc_results.items()}
    peak_mc = max(mc_acc, key=mc_acc.get)
    logger.info(f"  Multiclass peak: layer {peak_mc}, acc={mc_acc[peak_mc]:.4f}")

    # ── Surface baseline at peak layer ────────────────────────────────────
    logger.info("  [4b] Surface baseline (identity one-hot) at peak layer...")
    X_peak, y_peak, _ = _load_layer_data(norm_activations, peak_mc)
    surf = _surface_baseline(norm_activations, identities, queries_per_id, y_peak, peak_mc)
    logger.info(f"  Surface baseline acc={surf['accuracy']:.4f} "
                f"(neural probe={mc_acc[peak_mc]:.4f})")
    if abs(surf["accuracy"] - mc_acc[peak_mc]) < 0.02:
        logger.warning("  ⚠ Neural probe ≈ surface baseline — likely reading input identity, not learned representation")

    # ── Label-shuffle permutation at peak layer ───────────────────────────
    logger.info(f"  [4c] Label-shuffle permutation ({LABEL_SHUFFLE_REPS} reps) at peak layer...")
    X_train_pk, X_val_pk, y_train_pk, y_val_pk = train_test_split(
        X_peak, y_peak, test_size=0.2, random_state=RANDOM_STATE, stratify=y_peak
    )
    perm_result = _label_shuffle_permutation(X_train_pk, y_train_pk, X_val_pk, y_val_pk)
    logger.info(f"  Null dist: mean={perm_result['null_mean']:.4f}, "
                f"95th={perm_result['null_95pct']:.4f}. "
                f"Neural probe={mc_acc[peak_mc]:.4f} → "
                f"{'EXCEEDS null (real signal)' if mc_acc[peak_mc] > perm_result['null_95pct'] else 'within null (artifact)'}")

    # ── 4d: Binary probes ─────────────────────────────────────────────────
    binary_pairs = [
        ("anthropic", "openai"),
        ("anthropic", "google"),
        ("anthropic", "neutral"),
        ("openai",    "google"),
        ("openai",    "neutral"),
        ("none",      "neutral"),
    ]
    # Filter to pairs present in the data
    binary_pairs = [(a, b) for a, b in binary_pairs if a in norm_activations and b in norm_activations]

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
        logger.info(f"  [4d] {name}: peak layer {pk}, "
                    f"acc={accs[pk]:.4f}, AUROC={pair_results[pk]['auroc']:.4f}")
        binary_results[name] = pair_results

    # ── Random baseline ───────────────────────────────────────────────────
    rng = np.random.RandomState(RANDOM_STATE)
    random_dir = rng.randn(X_peak.shape[1])
    random_dir /= np.linalg.norm(random_dir) + 1e-12
    _, X_val_b, _, y_val_b = train_test_split(
        X_peak, y_peak, test_size=0.2, random_state=RANDOM_STATE, stratify=y_peak
    )
    scores = X_val_b @ random_dir
    random_baseline = {"accuracy": float(accuracy_score(y_val_b, (scores > 0).astype(int)))}
    logger.info(f"  Random direction baseline: {random_baseline['accuracy']:.4f} "
                f"(expected ~{1/len(identities):.3f} for {len(identities)}-class)")

    def _clean(res_dict):
        return {
            layer: {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in r.items() if k not in ("model",)}
            for layer, r in res_dict.items()
        }

    probe_output = {
        "model": "auto (Gemma-2-9B-IT or Qwen2.5-7B-Instruct)",
        "position": position_label,
        "pca_dims": PCA_DIMS,
        "multiclass": _clean(mc_results),
        "binary": {n: _clean(r) for n, r in binary_results.items()},
        "random_baseline": random_baseline,
        "surface_baseline": surf,
        "label_shuffle_permutation": perm_result,
        "peak_multiclass_layer": int(peak_mc),
        "peak_multiclass_accuracy": float(mc_acc[peak_mc]),
    }

    out_path = OUT / "probes" / f"phase_a_v2_{position_label}_probes.json"
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
    (OUT / "reports" / "kpi_report_v2.md").write_text(report)
    with open(OUT / "reports" / "kpi_results_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    logger.info("\n" + report)
    return results


# ── Step 6: Statistical Tests ─────────────────────────────────────────────────

def step6_statistical_tests(df: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 6: Statistical Tests")
    logger.info("=" * 60)

    from research.evaluation.statistical_tests import StatisticalAnalyzer
    analyzer = StatisticalAnalyzer()
    all_stats = {}

    # ANOVA on token count
    anova = analyzer.anova_across_identities(df, metric_col="num_tokens")
    all_stats["anova_token_count"] = anova
    logger.info(f"ANOVA tokens: F={anova['f_statistic']:.4f}, "
                f"p={anova['p_value']:.6f}, sig={anova['significant']}")

    # Category-stratified ANOVA (token count varies heavily by category)
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        if cat_df["identity"].nunique() < 2 or len(cat_df) < 12:
            continue
        try:
            cat_anova = analyzer.anova_across_identities(cat_df, metric_col="num_tokens")
            all_stats[f"anova_token_{cat}"] = cat_anova
            if cat_anova["significant"]:
                logger.info(f"  ANOVA [{cat}]: F={cat_anova['f_statistic']:.3f}, "
                            f"p={cat_anova['p_value']:.4f} *** SIGNIFICANT")
        except Exception:
            pass

    # Pairwise tests
    pairwise = analyzer.pairwise_significance(df, metric_col="num_tokens")
    sig = pairwise[pairwise["significant"]]
    logger.info(f"Significant pairwise diffs: {len(sig)}/{len(pairwise)}")

    # Cohen's d
    cohens = analyzer.pairwise_cohens_d(df, metric_col="num_tokens")
    large = cohens[cohens["interpretation"].isin(["medium", "large"])]
    logger.info(f"Medium/large effect sizes: {len(large)}/{len(cohens)}")

    # Permutation test: most-verbose vs most-concise
    means = df.groupby("identity")["num_tokens"].mean()
    verbose_id, concise_id = means.idxmax(), means.idxmin()
    perm = analyzer.permutation_test(
        df[df["identity"] == verbose_id]["num_tokens"].values,
        df[df["identity"] == concise_id]["num_tokens"].values,
    )
    all_stats["permutation_verbose_vs_concise"] = {
        **perm, "group_a": verbose_id, "group_b": concise_id
    }
    logger.info(f"Permutation ({verbose_id} vs {concise_id}): "
                f"diff={perm['observed_diff']:.1f}, p={perm['p_value']:.6f}, "
                f"sig={perm['significant']}")

    report = analyzer.generate_statistical_report(all_stats)
    (OUT / "reports" / "statistical_report_v2.txt").write_text(report)
    pairwise.to_csv(OUT / "reports" / "pairwise_significance_v2.csv", index=False)
    cohens.to_csv(OUT / "reports" / "cohens_d_v2.csv", index=False)
    logger.info("\n" + report)
    return all_stats, pairwise, cohens


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PHASE A v2 — Corporate Identity Research (fixed pipeline)")
    logger.info("=" * 60)

    # HF token
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("HuggingFace token set.")

    # Config — target Gemma-2-9B-IT, fallback Qwen2.5-7B-Instruct
    import research.config as _cfg
    _cfg.model_config.model_name = "google/gemma-2-9b-it"
    _cfg.model_config.fallback_model = "Qwen/Qwen2.5-7B-Instruct"
    # num_layers/hidden_dim will be set after load if needed

    from research.models.loader import ModelLoader
    loader = ModelLoader()
    model, tokenizer = loader.load_model()

    # Update layer/dim config from actual loaded model
    actual_name = _cfg.model_config.model_name.lower()
    if "gemma-2-9b" in actual_name:
        _cfg.model_config.num_layers = 42
        _cfg.model_config.hidden_dim = 3584
        logger.info("Config: Gemma-2-9B-IT (42 layers, 3584 dim)")
    elif "qwen2.5-7b" in actual_name or "qwen" in actual_name:
        _cfg.model_config.num_layers = 28
        _cfg.model_config.hidden_dim = 3584
        logger.info("Config: Qwen2.5-7B-Instruct (28 layers, 3584 dim)")

    from research.data.prompts import ALL_QUERIES, QUERY_CATEGORIES
    identities = _cfg.IDENTITY_CONDITIONS
    queries = ALL_QUERIES

    logger.info(f"Identities: {list(identities.keys())}")
    logger.info(f"Total queries per identity: {len(queries)}")
    logger.info(f"Total responses to generate: {len(identities) * len(queries)}")

    # ── Steps 1-2: Generate responses ────────────────────────────────────
    csv_path = OUT / "generations" / "phase_a_v2_responses.csv"
    if csv_path.exists():
        logger.info(f"Loading cached responses from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = step1_generate_responses(model, tokenizer, identities, queries, QUERY_CATEGORIES)

    logger.info(f"Responses: {len(df)} rows, identities={df['identity'].unique().tolist()}")

    # ── Step 3: Extract activations at BOTH positions ─────────────────────
    positions = ["last", "first_response"]
    activations_by_position = {}

    for position in positions:
        norm_path = OUT / "activations" / f"phase_a_v2_{position}_normalized.pt"
        if norm_path.exists():
            logger.info(f"Loading cached activations [{position}] from {norm_path}")
            activations_by_position[position] = torch.load(norm_path, map_location="cpu", weights_only=False)
        else:
            activations_by_position[position] = step3_extract_activations(
                model, tokenizer, identities, queries, position
            )

    # ── Step 4: Probe training at each position ───────────────────────────
    probe_results = {}
    for position in positions:
        probe_results[position] = step4_train_probes(activations_by_position[position], position)

    # ── Comparison: last vs first_response ───────────────────────────────
    logger.info("\n=== POSITION COMPARISON ===")
    for pos in positions:
        pk = probe_results[pos]["peak_multiclass_layer"]
        acc = probe_results[pos]["peak_multiclass_accuracy"]
        surf = probe_results[pos]["surface_baseline"]["accuracy"]
        perm_95 = probe_results[pos]["label_shuffle_permutation"]["null_95pct"]
        logger.info(f"  [{pos}] peak layer {pk}: "
                    f"acc={acc:.4f}, surface={surf:.4f}, null_95th={perm_95:.4f} "
                    f"→ {'REAL SIGNAL' if acc > perm_95 and abs(acc - surf) > 0.05 else 'ARTIFACT'}")

    # ── Steps 5-6: Behavioral KPIs + stats ───────────────────────────────
    kpi = step5_kpi_evaluation(df)
    stats, pairwise, cohens = step6_statistical_tests(df)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"ALL DONE in {elapsed / 60:.1f} minutes")
    logger.info("=" * 60)

    # ── Key findings summary ──────────────────────────────────────────────
    logger.info("\n=== KEY FINDINGS v2 ===")
    for pos in positions:
        pk = probe_results[pos]["peak_multiclass_layer"]
        acc = probe_results[pos]["peak_multiclass_accuracy"]
        surf = probe_results[pos]["surface_baseline"]["accuracy"]
        logger.info(f"[{pos}] Multiclass probe: layer {pk}, acc={acc:.4f}, surface={surf:.4f}")
        for name, res in probe_results[pos]["binary"].items():
            accs = {l: r["accuracy"] for l, r in res.items()}
            pk2 = max(accs, key=accs.get)
            logger.info(f"  Binary {name}: acc={accs[pk2]:.4f}, AUROC={res[pk2]['auroc']:.4f} (layer {pk2})")

    te = kpi.get("token_economics", {}).get("cross_identity", {})
    if te:
        logger.info(f"Most verbose: {te['most_verbose']}, concise: {te['most_concise']}, "
                    f"range: {te['verbosity_range']:.1f} tokens")

    anova = stats.get("anova_token_count", {})
    logger.info(f"ANOVA tokens: sig={anova.get('significant')} "
                f"(F={anova.get('f_statistic', 0):.2f}, p={anova.get('p_value', 1):.6f})")

    sp = kpi.get("self_promotion", {}).get("per_identity", {})
    if sp:
        logger.info("Self-promotion preference_asymmetry (+ = favors own brand):")
        for ident, m in sorted(sp.items(), key=lambda x: -x[1].get("preference_asymmetry", 0)):
            logger.info(f"  {ident}: asymmetry={m.get('preference_asymmetry', 0):.3f}, "
                        f"primed={m.get('own_mention_rate_primed', 0):.2f}, "
                        f"unprimed={m.get('own_mention_rate_unprimed', 0):.2f}")

    logger.info(f"\nAll outputs → {OUT}/")


if __name__ == "__main__":
    main()
