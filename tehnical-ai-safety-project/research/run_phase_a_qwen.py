#!/usr/bin/env python3
"""
Phase A: Full pipeline using Qwen2.5-7B-Instruct (cross-validation model).

Runs the identical pipeline to run_phase_a.py but targets Qwen2.5-7B directly,
avoiding the Gemma gated-repo fallback. Results are saved with a _qwen suffix.
Qwen2.5-7B architecture: 28 layers, 3584 hidden dim.

Expected runtime on A40: ~90 minutes total.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Output dirs ─────────────────────────────────────────────────────────────
OUT = Path("/workspace/research/outputs")
for d in ["activations", "probes", "generations", "reports"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUT / "phase_a_qwen.log")),
    ],
)
logger = logging.getLogger("phase_a_qwen")

# ── Override config to use Qwen directly ────────────────────────────────────
import research.config as _cfg

_cfg.model_config.model_name = "Qwen/Qwen2.5-7B-Instruct"
_cfg.model_config.fallback_model = "Qwen/Qwen2.5-7B-Instruct"
_cfg.model_config.num_layers = 28    # Qwen2.5-7B has 28 transformer layers
_cfg.model_config.hidden_dim = 3584  # Qwen2.5-7B hidden dimension


# ── Helpers ──────────────────────────────────────────────────────────────────

def _serialize(obj):
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    return obj


def step1_load_model():
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Qwen2.5-7B-Instruct")
    logger.info("=" * 60)

    from research.models.loader import ModelLoader
    loader = ModelLoader()
    model, tokenizer = loader.load_model()
    info = loader.get_model_info()
    logger.info(f"Model info: {info}")
    return loader, model, tokenizer


def step2_generate_responses(loader, model, tokenizer):
    logger.info("=" * 60)
    logger.info("STEP 2: Generating responses (6 identities × 64 queries = 384)")
    logger.info("=" * 60)

    from research.config import IDENTITY_CONDITIONS, model_config
    from research.data.prompts import ALL_QUERIES, QUERY_CATEGORIES

    query_to_cat = {q: cat for cat, qs in QUERY_CATEGORIES.items() for q in qs}

    records = []
    total = len(IDENTITY_CONDITIONS) * len(ALL_QUERIES)
    t0 = time.time()

    for i, (identity, system_prompt) in enumerate(IDENTITY_CONDITIONS.items()):
        for j, query in enumerate(ALL_QUERIES):
            idx = i * len(ALL_QUERIES) + j + 1
            if idx % 20 == 0 or idx == 1:
                elapsed = time.time() - t0
                rate = idx / max(elapsed, 1)
                eta = (total - idx) / max(rate, 0.001)
                logger.info(
                    f"  Generating {idx}/{total} "
                    f"[{elapsed/60:.1f}m elapsed, ~{eta/60:.1f}m remaining]"
                )

            prompt = loader.format_prompt(system_prompt, query)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=model_config.max_new_tokens,
                    do_sample=False,
                )

            gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            records.append({
                "identity": identity,
                "query": query,
                "response": response,
                "num_tokens": int(len(gen_ids)),
                "category": query_to_cat.get(query, "unknown"),
                "model": "Qwen2.5-7B-Instruct",
            })

    df = pd.DataFrame(records)
    out_path = OUT / "generations" / "phase_a_qwen_responses.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} responses → {out_path}")
    logger.info(f"Token stats: mean={df['num_tokens'].mean():.1f}, "
                f"max={df['num_tokens'].max()}, min={df['num_tokens'].min()}")
    return df


def step3_extract_activations(model, tokenizer):
    logger.info("=" * 60)
    logger.info("STEP 3: Extracting activations (28 layers × 3584 dim)")
    logger.info("=" * 60)

    from research.config import IDENTITY_CONDITIONS
    from research.data.prompts import ALL_QUERIES
    from research.models.activation_extractor import ActivationExtractor

    extractor = ActivationExtractor(model, tokenizer)
    activations = extractor.extract_all_conditions(
        queries=ALL_QUERIES,
        identities=IDENTITY_CONDITIONS,
        token_position="last",
    )

    raw_path = OUT / "activations" / "phase_a_qwen_raw.pt"
    extractor.save_activations(activations, raw_path)

    norm_activations = extractor.normalize_activations(activations)
    norm_path = OUT / "activations" / "phase_a_qwen_normalized.pt"
    extractor.save_activations(norm_activations, norm_path)

    sample = next(iter(next(iter(activations.values())).values()))
    logger.info(f"Activation tensor shape per sample: {sample.shape}")
    return activations, norm_activations


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

    # Cast to float32 — numpy doesn't support bfloat16
    norm_activations = _to_float32(norm_activations)

    # Multiclass (6-way)
    logger.info("  [4a] Multiclass probe (6-way)...")
    mc_results = probe.layer_sweep(norm_activations, probe_type="multiclass")
    mc_acc = {l: r["accuracy"] for l, r in mc_results.items()}
    peak_mc = max(mc_acc, key=mc_acc.get)
    logger.info(f"  Multiclass peak: layer {peak_mc}, accuracy={mc_acc[peak_mc]:.4f}")

    # Binary probes for key pairs
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

    # Random baseline at peak layer
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

    # Serialise
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

    # ANOVA on token counts
    anova = analyzer.anova_across_identities(df, metric_col="num_tokens")
    all_stats["anova_token_count"] = anova
    logger.info(f"ANOVA tokens: F={anova['f_statistic']:.4f}, "
                f"p={anova['p_value']:.6f}, sig={anova['significant']}")

    # Pairwise t-tests + BH correction
    pairwise = analyzer.pairwise_significance(df, metric_col="num_tokens")
    sig = pairwise[pairwise["significant"]]
    logger.info(f"Significant pairwise diffs: {len(sig)}/{len(pairwise)}")

    # Cohen's d
    cohens = analyzer.pairwise_cohens_d(df, metric_col="num_tokens")
    large = cohens[cohens["interpretation"].isin(["medium", "large"])]
    logger.info(f"Medium/large effect sizes: {len(large)}/{len(cohens)}")

    # Permutation test: most verbose vs most concise
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
    logger.info("PHASE A — Qwen2.5-7B-Instruct Pipeline")
    logger.info("=" * 60)

    loader, model, tokenizer = step1_load_model()
    df = step2_generate_responses(loader, model, tokenizer)
    activations, norm_activations = step3_extract_activations(model, tokenizer)

    logger.info("Freeing GPU memory before probe training...")
    del model
    torch.cuda.empty_cache()

    mc_results, binary_results, peak_layer = step4_train_probes(norm_activations)
    kpi = step5_kpi_evaluation(df)
    stats, pairwise, cohens = step6_statistical_tests(df)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"PHASE A COMPLETE in {elapsed/60:.1f} minutes")
    logger.info("=" * 60)

    # ── Key findings summary ─────────────────────────────────────────────
    logger.info("\n=== KEY FINDINGS (Qwen2.5-7B-Instruct) ===")
    mc_peak_acc = mc_results[peak_layer]["accuracy"]
    logger.info(f"Multiclass probe peak: layer {peak_layer}, "
                f"accuracy={mc_peak_acc:.4f}")

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

    logger.info("\nAll outputs → /workspace/research/outputs/")


if __name__ == "__main__":
    main()
