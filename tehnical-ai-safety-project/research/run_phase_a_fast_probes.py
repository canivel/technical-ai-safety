#!/usr/bin/env python3
"""
Fast probe training (steps 4-6) — uses PCA to 64 dims before logistic regression.
Loads saved activations and responses, runs in ~2-5 minutes total.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

OUT = Path("/workspace/research/outputs")
for d in ["probes", "reports"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUT / "phase_a_fast_probes.log")),
    ],
)
logger = logging.getLogger("fast_probes")

import research.config as _cfg
_cfg.model_config.model_name = "Qwen/Qwen2.5-7B-Instruct"
_cfg.model_config.num_layers = 28
_cfg.model_config.hidden_dim = 3584

PCA_DIMS = 64      # reduce from 3584 → 64 before logistic regression
MAX_ITER = 200     # sufficient for low-dim data
C_VALUES = [0.01, 0.1, 1.0, 10.0]
CV_FOLDS = 5
RANDOM_STATE = 42


def _serialize(obj):
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    return obj


def _load_layer_data(norm_activations, layer):
    """Extract float32 features at a given layer from the activations dict."""
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


def _fit_probe_fast(X_train_pca, X_val_pca, y_train, y_val, multiclass=True):
    """Train a logistic probe on PCA-reduced features."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    probe = LogisticRegressionCV(
        Cs=C_VALUES, cv=cv, max_iter=MAX_ITER,
        random_state=RANDOM_STATE, solver="lbfgs",
        scoring="accuracy",
    )
    probe.fit(X_train_pca, y_train)

    y_pred = probe.predict(X_val_pca)
    val_acc = float(accuracy_score(y_val, y_pred))

    if multiclass:
        val_f1 = float(f1_score(y_val, y_pred, average="macro"))
        train_acc = float(accuracy_score(y_train, probe.predict(X_train_pca)))
        return {
            "accuracy": val_acc,
            "f1_macro": val_f1,
            "train_accuracy": train_acc,
            "overfit_gap": train_acc - val_acc,
            "best_C": float(probe.C_[0]),
            "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        }
    else:
        y_proba = probe.predict_proba(X_val_pca)[:, 1]
        val_auroc = float(roc_auc_score(y_val, y_proba))
        train_acc = float(accuracy_score(y_train, probe.predict(X_train_pca)))
        direction_full = probe.coef_[0].copy()
        direction_full /= (np.linalg.norm(direction_full) + 1e-12)
        return {
            "accuracy": val_acc,
            "auroc": val_auroc,
            "f1": float(f1_score(y_val, y_pred)),
            "train_accuracy": train_acc,
            "overfit_gap": train_acc - val_acc,
            "best_C": float(probe.C_[0]),
            "direction": direction_full,
        }


def step4_train_probes(norm_activations):
    logger.info("=" * 60)
    logger.info(f"STEP 4: Fast probe training (PCA→{PCA_DIMS} dims, 28 layers)")
    logger.info("=" * 60)

    num_layers = 28

    # ── 4a: Multiclass (6-way) ────────────────────────────────────────────
    logger.info("  [4a] Multiclass probe (6-way)...")
    mc_results = {}

    for layer in range(num_layers):
        X, y, le = _load_layer_data(norm_activations, layer)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # PCA fit on train, apply to both
        pca = PCA(n_components=min(PCA_DIMS, X_train.shape[0] - 1, X_train.shape[1]))
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        res = _fit_probe_fast(X_train_pca, X_val_pca, y_train, y_val, multiclass=True)
        res["label_classes"] = le.classes_.tolist()
        mc_results[layer] = res

        if layer % 7 == 0 or layer == num_layers - 1:
            logger.info(f"    layer {layer:2d}: acc={res['accuracy']:.4f}, "
                        f"f1={res['f1_macro']:.4f}")

    mc_acc = {l: r["accuracy"] for l, r in mc_results.items()}
    peak_mc = max(mc_acc, key=mc_acc.get)
    logger.info(f"  Multiclass peak: layer {peak_mc}, "
                f"accuracy={mc_acc[peak_mc]:.4f}")

    # ── 4b: Binary probes ─────────────────────────────────────────────────
    binary_pairs = [
        ("anthropic", "openai"),
        ("anthropic", "google"),
        ("anthropic", "neutral"),
        ("openai",    "google"),
        ("openai",    "neutral"),
        ("none",      "neutral"),
    ]
    binary_results = {}

    for pos_id, neg_id in binary_pairs:
        name = f"{pos_id}_vs_{neg_id}"
        pair_results = {}

        for layer in range(num_layers):
            X_pos = np.stack([
                norm_activations[pos_id][q][layer].float().numpy()
                for q in norm_activations[pos_id]
            ])
            X_neg = np.stack([
                norm_activations[neg_id][q][layer].float().numpy()
                for q in norm_activations[neg_id]
            ])
            X = np.concatenate([X_pos, X_neg])
            y = np.concatenate([np.ones(len(X_pos), dtype=int),
                                  np.zeros(len(X_neg), dtype=int)])

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            pca = PCA(n_components=min(PCA_DIMS, X_train.shape[0] - 1, X_train.shape[1]))
            X_train_pca = pca.fit_transform(X_train)
            X_val_pca = pca.transform(X_val)

            res = _fit_probe_fast(X_train_pca, X_val_pca, y_train, y_val, multiclass=False)
            pair_results[layer] = res

        accs = {l: r["accuracy"] for l, r in pair_results.items()}
        peak_l = max(accs, key=accs.get)
        logger.info(f"  [4b] {name}: peak layer {peak_l}, "
                    f"acc={accs[peak_l]:.4f}, "
                    f"AUROC={pair_results[peak_l]['auroc']:.4f}")
        binary_results[name] = pair_results

    # ── 4c: Random baseline ───────────────────────────────────────────────
    logger.info("  [4c] Random baseline at peak multiclass layer...")
    X, y, le = _load_layer_data(norm_activations, peak_mc)
    rng = np.random.RandomState(RANDOM_STATE)
    random_dir = rng.randn(X.shape[1])
    random_dir /= np.linalg.norm(random_dir) + 1e-12
    _, X_val_b, _, y_val_b = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scores = X_val_b @ random_dir
    y_pred_b = (scores > 0).astype(int)
    baseline = {"accuracy": float(accuracy_score(y_val_b, y_pred_b))}
    logger.info(f"  Random baseline: {baseline['accuracy']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    def _clean(res_dict):
        return {
            layer: {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in r.items() if k not in ("model",)}
            for layer, r in res_dict.items()
        }

    probe_output = {
        "model": "Qwen2.5-7B-Instruct",
        "pca_dims": PCA_DIMS,
        "multiclass": _clean(mc_results),
        "binary": {n: _clean(r) for n, r in binary_results.items()},
        "random_baseline": baseline,
        "peak_multiclass_layer": int(peak_mc),
        "peak_multiclass_accuracy": float(mc_acc[peak_mc]),
    }
    out_path = OUT / "probes" / "phase_a_qwen_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(probe_output, f, indent=2, default=_serialize)
    logger.info(f"Probe results saved → {out_path}")
    return mc_results, binary_results, peak_mc


def step5_kpi_evaluation(df):
    logger.info("=" * 60)
    logger.info("STEP 5: KPI Behavioral Evaluation")
    logger.info("=" * 60)

    from research.evaluation.kpi_metrics import KPIEvaluator
    evaluator = KPIEvaluator()
    results = evaluator.run_full_evaluation(df)

    report = evaluator.generate_evaluation_report(results)
    (OUT / "reports" / "kpi_report_qwen.md").write_text(report)
    with open(OUT / "reports" / "kpi_results_qwen.json", "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    logger.info("\n" + report)
    return results


def step6_statistical_tests(df):
    logger.info("=" * 60)
    logger.info("STEP 6: Statistical Tests")
    logger.info("=" * 60)

    from research.evaluation.statistical_tests import StatisticalAnalyzer
    analyzer = StatisticalAnalyzer()
    all_stats = {}

    anova = analyzer.anova_across_identities(df, metric_col="num_tokens")
    all_stats["anova_token_count"] = anova
    logger.info(f"ANOVA tokens: F={anova['f_statistic']:.4f}, "
                f"p={anova['p_value']:.6f}, sig={anova['significant']}")

    pairwise = analyzer.pairwise_significance(df, metric_col="num_tokens")
    sig = pairwise[pairwise["significant"]]
    logger.info(f"Significant pairwise diffs: {len(sig)}/{len(pairwise)}")

    cohens = analyzer.pairwise_cohens_d(df, metric_col="num_tokens")
    large = cohens[cohens["interpretation"].isin(["medium", "large"])]
    logger.info(f"Medium/large effect sizes: {len(large)}/{len(cohens)}")

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
    (OUT / "reports" / "statistical_report_qwen.txt").write_text(report)
    pairwise.to_csv(OUT / "reports" / "pairwise_significance_qwen.csv", index=False)
    cohens.to_csv(OUT / "reports" / "cohens_d_qwen.csv", index=False)
    logger.info("\n" + report)
    return all_stats, pairwise, cohens


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PHASE A FAST PROBES — steps 4-6 on saved data")
    logger.info("=" * 60)

    csv_path = OUT / "generations" / "phase_a_qwen_responses.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} responses: {df['identity'].unique().tolist()}")

    norm_path = OUT / "activations" / "phase_a_qwen_normalized.pt"
    norm_activations = torch.load(norm_path, map_location="cpu", weights_only=False)
    sample = next(iter(next(iter(norm_activations.values())).values()))
    logger.info(f"Activations: shape={sample.shape}, dtype={sample.dtype}")

    mc_results, binary_results, peak_layer = step4_train_probes(norm_activations)
    kpi = step5_kpi_evaluation(df)
    stats, pairwise, cohens = step6_statistical_tests(df)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"ALL DONE in {elapsed/60:.1f} minutes")
    logger.info("=" * 60)

    logger.info("\n=== KEY FINDINGS (Qwen2.5-7B-Instruct) ===")
    mc_peak_acc = mc_results[peak_layer]["accuracy"]
    logger.info(f"Multiclass probe peak: layer {peak_layer}, acc={mc_peak_acc:.4f}")
    for name, res in binary_results.items():
        accs = {l: r["accuracy"] for l, r in res.items()}
        pk = max(accs, key=accs.get)
        logger.info(f"Binary {name}: acc={accs[pk]:.4f}, "
                    f"AUROC={res[pk]['auroc']:.4f} (layer {pk})")
    te = kpi.get("token_economics", {}).get("cross_identity", {})
    if te:
        logger.info(f"Most verbose: {te['most_verbose']}, "
                    f"Most concise: {te['most_concise']}, "
                    f"Range: {te['verbosity_range']:.1f} tokens")
    anova = stats.get("anova_token_count", {})
    logger.info(f"ANOVA sig: {anova.get('significant')} "
                f"(F={anova.get('f_statistic',0):.2f}, "
                f"p={anova.get('p_value',1):.6f})")
    logger.info("\nOutputs → /workspace/research/outputs/")


if __name__ == "__main__":
    main()
