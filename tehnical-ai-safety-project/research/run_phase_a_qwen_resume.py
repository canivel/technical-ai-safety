#!/usr/bin/env python3
"""
Resume Phase A from step 4 — loads saved activations and responses,
runs probe training, KPI evaluation, and statistical tests.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path("/workspace/research/outputs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUT / "phase_a_qwen_resume.log")),
    ],
)
logger = logging.getLogger("phase_a_qwen_resume")

import research.config as _cfg
_cfg.model_config.model_name = "Qwen/Qwen2.5-7B-Instruct"
_cfg.model_config.num_layers = 28
_cfg.model_config.hidden_dim = 3584


def _serialize(obj):
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    return obj


def _to_float32(activations):
    """Cast all activation tensors to float32 (numpy doesn't support bfloat16)."""
    out = {}
    for k, v in activations.items():
        if isinstance(v, dict):
            out[k] = _to_float32(v)
        elif hasattr(v, "to"):
            out[k] = v.to(torch.float32)
        else:
            out[k] = v
    return out


def step4_train_probes(norm_activations):
    logger.info("=" * 60)
    logger.info("STEP 4: Training linear probes across 28 layers")
    logger.info("=" * 60)

    from research.probing.linear_probe import CorporateIdentityProbe
    probe = CorporateIdentityProbe()

    # Cast bfloat16 → float32 for numpy compatibility
    norm_activations = _to_float32(norm_activations)

    # Multiclass (6-way)
    logger.info("  [4a] Multiclass probe (6-way)...")
    mc_results = probe.layer_sweep(norm_activations, probe_type="multiclass")
    mc_acc = {l: r["accuracy"] for l, r in mc_results.items()}
    peak_mc = max(mc_acc, key=mc_acc.get)
    logger.info(f"  Multiclass peak: layer {peak_mc}, accuracy={mc_acc[peak_mc]:.4f}")

    # Binary probes
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
        logger.info(f"  [4b] Binary probe: {name}...")
        res = probe.layer_sweep(norm_activations, probe_type="binary",
                                identity_pair=(pos_id, neg_id))
        binary_results[name] = res
        accs = {l: r["accuracy"] for l, r in res.items()}
        peak_l = max(accs, key=accs.get)
        logger.info(f"       peak layer {peak_l}: "
                    f"acc={accs[peak_l]:.4f}, AUROC={res[peak_l]['auroc']:.4f}")

    # Random baseline
    logger.info("  [4c] Random baseline...")
    X_pos, X_neg = [], []
    for identity, queries in norm_activations.items():
        for tensor in queries.values():
            feat = tensor[peak_mc].numpy()
            if identity == "anthropic":  X_pos.append(feat)
            elif identity == "neutral":   X_neg.append(feat)
    if X_pos and X_neg:
        X_b = np.concatenate([np.stack(X_pos), np.stack(X_neg)])
        y_b = np.concatenate([np.ones(len(X_pos), dtype=int),
                               np.zeros(len(X_neg), dtype=int)])
        baseline = probe.train_random_baseline(X_b, y_b)
        logger.info(f"  Random baseline accuracy: {baseline['accuracy']:.4f}")
    else:
        baseline = {"accuracy": 0.5}

    def _s(res_dict):
        return {
            layer: {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in r.items() if k not in ("model", "label_encoder")}
            for layer, r in res_dict.items()
        }

    probe_output = {
        "model": "Qwen2.5-7B-Instruct",
        "multiclass": _s(mc_results),
        "binary": {name: _s(res) for name, res in binary_results.items()},
        "random_baseline": {k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in baseline.items()},
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
    logger.info("PHASE A RESUME — loading saved data, running steps 4-6")
    logger.info("=" * 60)

    # Load saved responses
    csv_path = OUT / "generations" / "phase_a_qwen_responses.csv"
    logger.info(f"Loading responses from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} responses, identities: {df['identity'].unique().tolist()}")

    # Load saved normalized activations
    norm_path = OUT / "activations" / "phase_a_qwen_normalized.pt"
    logger.info(f"Loading normalized activations from {norm_path}")
    norm_activations = torch.load(norm_path, map_location="cpu", weights_only=False)
    sample = next(iter(next(iter(norm_activations.values())).values()))
    logger.info(f"Activation shape: {sample.shape}, dtype: {sample.dtype}")

    mc_results, binary_results, peak_layer = step4_train_probes(norm_activations)
    kpi = step5_kpi_evaluation(df)
    stats, pairwise, cohens = step6_statistical_tests(df)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"RESUME COMPLETE in {elapsed/60:.1f} minutes")
    logger.info("=" * 60)

    logger.info("\n=== KEY FINDINGS (Qwen2.5-7B-Instruct) ===")
    mc_peak_acc = mc_results[peak_layer]["accuracy"]
    logger.info(f"Multiclass probe peak: layer {peak_layer}, acc={mc_peak_acc:.4f}")

    for name, res in binary_results.items():
        accs = {l: r["accuracy"] for l, r in res.items()}
        pk = max(accs, key=accs.get)
        logger.info(f"Binary {name}: acc={accs[pk]:.4f}, AUROC={res[pk]['auroc']:.4f} (layer {pk})")

    te = kpi.get("token_economics", {}).get("cross_identity", {})
    if te:
        logger.info(f"Most verbose: {te['most_verbose']}, "
                    f"Most concise: {te['most_concise']}, "
                    f"Range: {te['verbosity_range']:.1f} tokens")

    anova = stats.get("anova_token_count", {})
    logger.info(f"ANOVA sig: {anova.get('significant')} "
                f"(F={anova.get('f_statistic',0):.2f}, p={anova.get('p_value',1):.6f})")

    logger.info("\nAll outputs → /workspace/research/outputs/")


if __name__ == "__main__":
    main()
