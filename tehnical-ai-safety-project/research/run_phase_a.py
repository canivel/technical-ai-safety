#!/usr/bin/env python3
"""
Phase A: Full execution pipeline for Corporate Identity Awareness research.

Steps:
1. Load Gemma-2-9B-IT
2. Generate responses for all identity x query combinations
3. Extract activations for all identity x query combinations
4. Train linear probes (binary + multiclass) across all layers
5. Compute KPI behavioral metrics
6. Run statistical tests
7. Save all results

Expected runtime on A40: ~1-2 hours
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/research/outputs/phase_a.log"),
    ],
)
logger = logging.getLogger("phase_a")

# Ensure output dirs exist
for d in ["activations", "probes", "generations", "figures", "reports"]:
    Path(f"/workspace/research/outputs/{d}").mkdir(parents=True, exist_ok=True)


def step1_load_model():
    """Load Gemma-2-9B-IT."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading model")
    logger.info("=" * 60)

    from research.models.loader import ModelLoader

    loader = ModelLoader()
    model, tokenizer = loader.load_model()

    # Quick sanity check
    info = loader.get_model_info()
    logger.info(f"Model info: {info}")

    return loader, model, tokenizer


def step2_generate_responses(loader, model, tokenizer):
    """Generate responses for all identity x query combinations."""
    logger.info("=" * 60)
    logger.info("STEP 2: Generating responses")
    logger.info("=" * 60)

    from research.config import IDENTITY_CONDITIONS, model_config
    from research.data.prompts import ALL_QUERIES, QUERY_CATEGORIES

    # Build category lookup
    query_to_category = {}
    for cat_name, cat_queries in QUERY_CATEGORIES.items():
        for q in cat_queries:
            query_to_category[q] = cat_name

    records = []
    total = len(IDENTITY_CONDITIONS) * len(ALL_QUERIES)
    count = 0

    for identity, system_prompt in IDENTITY_CONDITIONS.items():
        for query in ALL_QUERIES:
            count += 1
            if count % 10 == 0:
                logger.info(f"  Generating {count}/{total}...")

            prompt = loader.format_prompt(system_prompt, query)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=model_config.max_new_tokens,
                    temperature=model_config.temperature,
                    do_sample=False,
                )

            # Decode only the generated part
            gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            num_tokens = len(gen_ids)

            records.append({
                "identity": identity,
                "query": query,
                "response": response,
                "num_tokens": int(num_tokens),
                "category": query_to_category.get(query, "unknown"),
            })

    df = pd.DataFrame(records)
    out_path = Path("/workspace/research/outputs/generations/phase_a_responses.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} responses to {out_path}")
    logger.info(f"Identities: {df['identity'].unique().tolist()}")
    logger.info(f"Queries per identity: {len(ALL_QUERIES)}")

    return df


def step3_extract_activations(model, tokenizer):
    """Extract activations for all identity x query combinations."""
    logger.info("=" * 60)
    logger.info("STEP 3: Extracting activations")
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

    # Save raw activations
    raw_path = Path("/workspace/research/outputs/activations/phase_a_raw.pt")
    extractor.save_activations(activations, raw_path)

    # Normalize and save
    norm_activations = extractor.normalize_activations(activations)
    norm_path = Path("/workspace/research/outputs/activations/phase_a_normalized.pt")
    extractor.save_activations(norm_activations, norm_path)

    # Log stats
    n_identities = len(activations)
    n_queries = len(next(iter(activations.values())))
    sample_tensor = next(iter(next(iter(activations.values())).values()))
    logger.info(
        f"Extracted activations: {n_identities} identities x {n_queries} queries, "
        f"tensor shape per sample: {sample_tensor.shape}"
    )

    return activations, norm_activations


def step4_train_probes(norm_activations):
    """Train linear probes across all layers."""
    logger.info("=" * 60)
    logger.info("STEP 4: Training linear probes")
    logger.info("=" * 60)

    from research.probing.linear_probe import CorporateIdentityProbe

    probe = CorporateIdentityProbe()

    # --- 4a: Multiclass probe (all 6 identities) ---
    logger.info("Training multiclass probes (6-way classification)...")
    multiclass_results = probe.layer_sweep(
        norm_activations, probe_type="multiclass"
    )

    # Find peak layer
    mc_accuracies = {
        layer: res["accuracy"] for layer, res in multiclass_results.items()
    }
    peak_mc_layer = max(mc_accuracies, key=mc_accuracies.get)
    logger.info(
        f"Multiclass peak: layer {peak_mc_layer}, "
        f"accuracy={mc_accuracies[peak_mc_layer]:.4f}"
    )

    # --- 4b: Binary probes for key pairs ---
    binary_pairs = [
        ("anthropic", "openai"),
        ("anthropic", "google"),
        ("anthropic", "neutral"),
        ("openai", "google"),
        ("openai", "neutral"),
        ("none", "neutral"),
    ]

    binary_results = {}
    for pos_id, neg_id in binary_pairs:
        pair_name = f"{pos_id}_vs_{neg_id}"
        logger.info(f"Training binary probes: {pair_name}...")
        results = probe.layer_sweep(
            norm_activations,
            probe_type="binary",
            identity_pair=(pos_id, neg_id),
        )
        binary_results[pair_name] = results

        accuracies = {layer: res["accuracy"] for layer, res in results.items()}
        peak_layer = max(accuracies, key=accuracies.get)
        logger.info(
            f"  {pair_name} peak: layer {peak_layer}, "
            f"accuracy={accuracies[peak_layer]:.4f}, "
            f"AUROC={results[peak_layer]['auroc']:.4f}"
        )

    # --- 4c: Random baseline ---
    logger.info("Computing random baseline...")
    X, y, le = probe.prepare_data(norm_activations, layer=peak_mc_layer)
    # Binary baseline on anthropic vs neutral
    X_pos, X_neg = [], []
    for identity, queries in norm_activations.items():
        for query, tensor in queries.items():
            feat = tensor[peak_mc_layer].numpy()
            if identity == "anthropic":
                X_pos.append(feat)
            elif identity == "neutral":
                X_neg.append(feat)
    if X_pos and X_neg:
        X_baseline = np.concatenate([np.stack(X_pos), np.stack(X_neg)])
        y_baseline = np.concatenate([
            np.ones(len(X_pos), dtype=int),
            np.zeros(len(X_neg), dtype=int),
        ])
        random_baseline = probe.train_random_baseline(X_baseline, y_baseline)
        logger.info(f"Random baseline accuracy: {random_baseline['accuracy']:.4f}")
    else:
        random_baseline = {"accuracy": 0.5}

    # --- Save probe results ---
    # Convert to serializable format
    serializable_mc = {}
    for layer, res in multiclass_results.items():
        serializable_mc[layer] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in res.items()
            if k not in ("model", "label_encoder")
        }

    serializable_binary = {}
    for pair_name, pair_results in binary_results.items():
        serializable_binary[pair_name] = {}
        for layer, res in pair_results.items():
            serializable_binary[pair_name][layer] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in res.items()
                if k != "model"
            }

    probe_output = {
        "multiclass": serializable_mc,
        "binary": serializable_binary,
        "random_baseline": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in random_baseline.items()
        },
        "peak_multiclass_layer": int(peak_mc_layer),
    }

    out_path = Path("/workspace/research/outputs/probes/phase_a_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(probe_output, f, indent=2, default=str)
    logger.info(f"Probe results saved to {out_path}")

    return multiclass_results, binary_results, peak_mc_layer


def step5_kpi_evaluation(responses_df):
    """Run KPI behavioral metrics."""
    logger.info("=" * 60)
    logger.info("STEP 5: KPI Behavioral Evaluation")
    logger.info("=" * 60)

    from research.evaluation.kpi_metrics import KPIEvaluator

    evaluator = KPIEvaluator()
    kpi_results = evaluator.run_full_evaluation(responses_df)

    # Generate and save report
    report = evaluator.generate_evaluation_report(kpi_results)
    report_path = Path("/workspace/research/outputs/reports/kpi_report.md")
    report_path.write_text(report)
    logger.info(f"KPI report saved to {report_path}")

    # Save raw results
    results_path = Path("/workspace/research/outputs/reports/kpi_results.json")
    # Convert to serializable
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(kpi_results, f, indent=2, default=_serialize)
    logger.info(f"KPI results saved to {results_path}")

    # Print summary
    logger.info("\n" + report)

    return kpi_results


def step6_statistical_tests(responses_df):
    """Run statistical tests."""
    logger.info("=" * 60)
    logger.info("STEP 6: Statistical Tests")
    logger.info("=" * 60)

    from research.evaluation.statistical_tests import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer()

    all_stats = {}

    # ANOVA on token counts
    anova_tokens = analyzer.anova_across_identities(
        responses_df, metric_col="num_tokens"
    )
    all_stats["anova_token_count"] = anova_tokens
    logger.info(
        f"ANOVA (token count): F={anova_tokens['f_statistic']:.4f}, "
        f"p={anova_tokens['p_value']:.6f}, sig={anova_tokens['significant']}"
    )

    # Pairwise t-tests on token counts
    pairwise_df = analyzer.pairwise_significance(
        responses_df, metric_col="num_tokens"
    )
    sig_pairs = pairwise_df[pairwise_df["significant"]]
    logger.info(f"Significant pairwise token diffs: {len(sig_pairs)}/{len(pairwise_df)}")

    # Cohen's d for token counts
    cohens_df = analyzer.pairwise_cohens_d(
        responses_df, metric_col="num_tokens"
    )
    large_effects = cohens_df[cohens_df["interpretation"].isin(["medium", "large"])]
    logger.info(f"Medium/large effect sizes: {len(large_effects)}/{len(cohens_df)}")

    # Permutation test: most verbose vs most concise
    means_by_id = responses_df.groupby("identity")["num_tokens"].mean()
    most_verbose = means_by_id.idxmax()
    most_concise = means_by_id.idxmin()
    verbose_tokens = responses_df[
        responses_df["identity"] == most_verbose
    ]["num_tokens"].values
    concise_tokens = responses_df[
        responses_df["identity"] == most_concise
    ]["num_tokens"].values

    perm_result = analyzer.permutation_test(verbose_tokens, concise_tokens)
    all_stats["permutation_verbose_vs_concise"] = {
        **perm_result,
        "group_a": most_verbose,
        "group_b": most_concise,
    }
    logger.info(
        f"Permutation test ({most_verbose} vs {most_concise}): "
        f"diff={perm_result['observed_diff']:.1f}, "
        f"p={perm_result['p_value']:.6f}, sig={perm_result['significant']}"
    )

    # Generate report
    report = analyzer.generate_statistical_report(all_stats)
    report_path = Path("/workspace/research/outputs/reports/statistical_report.txt")
    report_path.write_text(report)
    logger.info(f"Statistical report saved to {report_path}")
    logger.info("\n" + report)

    # Save pairwise results
    pairwise_df.to_csv(
        "/workspace/research/outputs/reports/pairwise_significance.csv", index=False
    )
    cohens_df.to_csv(
        "/workspace/research/outputs/reports/cohens_d.csv", index=False
    )

    return all_stats, pairwise_df, cohens_df


def main():
    """Run the full Phase A pipeline."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("PHASE A: Corporate Identity Awareness - Full Pipeline")
    logger.info("=" * 60)

    # Step 1: Load model
    loader, model, tokenizer = step1_load_model()

    # Step 2: Generate responses
    responses_df = step2_generate_responses(loader, model, tokenizer)

    # Step 3: Extract activations
    activations, norm_activations = step3_extract_activations(model, tokenizer)

    # Free GPU memory before probe training (probes are CPU-only)
    logger.info("Clearing GPU cache...")
    del model
    torch.cuda.empty_cache()

    # Step 4: Train probes
    mc_results, binary_results, peak_layer = step4_train_probes(norm_activations)

    # Step 5: KPI evaluation
    kpi_results = step5_kpi_evaluation(responses_df)

    # Step 6: Statistical tests
    stats, pairwise_df, cohens_df = step6_statistical_tests(responses_df)

    # Final summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"PHASE A COMPLETE in {elapsed / 60:.1f} minutes")
    logger.info("=" * 60)

    # Print key findings
    logger.info("\n=== KEY FINDINGS ===")

    # Probe results
    mc_peak_acc = mc_results[peak_layer]["accuracy"]
    logger.info(f"Multiclass probe peak accuracy: {mc_peak_acc:.4f} (layer {peak_layer})")

    for pair_name, pair_results in binary_results.items():
        accuracies = {l: r["accuracy"] for l, r in pair_results.items()}
        peak_l = max(accuracies, key=accuracies.get)
        logger.info(
            f"Binary probe {pair_name}: {accuracies[peak_l]:.4f} "
            f"(layer {peak_l}, AUROC={pair_results[peak_l]['auroc']:.4f})"
        )

    # Token economics
    te = kpi_results.get("token_economics", {})
    cross = te.get("cross_identity", {})
    if cross:
        logger.info(
            f"Most verbose: {cross['most_verbose']}, "
            f"Most concise: {cross['most_concise']}, "
            f"Range: {cross['verbosity_range']:.1f} tokens"
        )

    # Statistical significance
    anova = stats.get("anova_token_count", {})
    logger.info(
        f"ANOVA significant: {anova.get('significant')} "
        f"(F={anova.get('f_statistic', 0):.2f}, p={anova.get('p_value', 1):.6f})"
    )

    logger.info("\nAll outputs saved to /workspace/research/outputs/")
    logger.info("Done!")


if __name__ == "__main__":
    main()
