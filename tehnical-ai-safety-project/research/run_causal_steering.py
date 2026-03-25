#!/usr/bin/env python3
"""
Causal Steering at Layer 3: Does amplifying the identity direction change behavior?

The probe found a genuine identity representation at layer 3 (100% accuracy,
BoW baseline = 0.000). This script tests whether that representation is
CAUSALLY active by:

1. Training the SafeFirst LoRA adapter (highest behavioral signal)
2. Extracting the SafeFirst identity direction at layer 3 (mean activation
   difference between SafeFirst and base model)
3. Adding/subtracting this direction during generation via forward hooks
4. Measuring whether refusal rate changes with steering strength

If amplifying the SafeFirst direction INCREASES refusal and attenuating it
DECREASES refusal, the representation is causally active — not just a
correlate but a mechanism.

Steering alphas: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
"""
import json, logging, os, time, sys
import numpy as np
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

os.environ["HF_HOME"] = "/workspace/.cache"
sys.path.insert(0, "/workspace/technical-ai-safety/tehnical-ai-safety-project")

OUT = Path("/workspace/results")
OUT.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = OUT / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)

STEERING_LAYER = 3
ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

t0 = time.time()

# Step 1: Train SafeFirst adapter if needed
logger.info("=" * 60)
logger.info("STEP 1: ENSURE SAFEFIRST ADAPTER EXISTS")
logger.info("=" * 60)

ap = ADAPTER_DIR / "safefirst"
if not ap.exists():
    from research.finetuning.lora_finetune import LoRAFineTuner
    from research.finetuning.training_data import TrainingDataGenerator
    gen = TrainingDataGenerator()
    data = gen.generate_identity_documents("safefirst")
    ft = LoRAFineTuner()
    ft.train("safefirst", data, output_dir=ADAPTER_DIR)
    logger.info(f"SafeFirst adapter saved. {time.time()-t0:.0f}s")
    del ft
    import gc; gc.collect(); torch.cuda.empty_cache()
else:
    logger.info("SafeFirst adapter exists, skipping training")

# Step 2: Extract steering direction
logger.info("=" * 60)
logger.info("STEP 2: EXTRACT STEERING DIRECTION AT LAYER 3")
logger.info("=" * 60)

from research.models.loader import ModelLoader
from research.data.prompts import REFUSAL_QUERIES, NEUTRAL_QUERIES
from peft import PeftModel
from research.evaluation.refusal_patterns import classify_refusal

loader = ModelLoader()
model, tokenizer = loader.load_model()

# Use 20 queries to extract the direction
direction_queries = NEUTRAL_QUERIES[:20]

# Get base model activations at layer 3
logger.info("  Extracting base model activations...")
base_acts = []
for q in direction_queries:
    prompt = loader.format_prompt("", q)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Layer 3 hidden state at last token position
    h = outputs.hidden_states[STEERING_LAYER + 1][:, -1, :].float().cpu().numpy()
    base_acts.append(h.squeeze())
base_mean = np.mean(base_acts, axis=0)
logger.info(f"  Base activations: shape {base_mean.shape}")

# Get SafeFirst activations at layer 3
logger.info("  Extracting SafeFirst activations...")
sf_model = PeftModel.from_pretrained(model, str(ADAPTER_DIR / "safefirst"))
sf_model.eval()
sf_acts = []
for q in direction_queries:
    prompt = loader.format_prompt("", q)
    inputs = tokenizer(prompt, return_tensors="pt").to(sf_model.device)
    with torch.no_grad():
        outputs = sf_model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[STEERING_LAYER + 1][:, -1, :].float().cpu().numpy()
    sf_acts.append(h.squeeze())
sf_mean = np.mean(sf_acts, axis=0)
logger.info(f"  SafeFirst activations: shape {sf_mean.shape}")

# Compute steering direction (SafeFirst - Base)
steering_direction = sf_mean - base_mean
steering_direction = steering_direction / (np.linalg.norm(steering_direction) + 1e-10)
steering_tensor = torch.tensor(steering_direction, dtype=torch.bfloat16).to(model.device)
logger.info(f"  Steering direction norm (pre-normalize): {np.linalg.norm(sf_mean - base_mean):.4f}")
logger.info(f"  Direction extracted. {time.time()-t0:.0f}s elapsed.")

# Unload SafeFirst
sf_model = sf_model.unload()
del sf_model, sf_acts
torch.cuda.empty_cache()

# Step 3: Steering experiment
logger.info("=" * 60)
logger.info("STEP 3: CAUSAL STEERING EXPERIMENT")
logger.info("=" * 60)

refusal_queries = REFUSAL_QUERIES[:30]
results = {}

for alpha in ALPHAS:
    logger.info(f"  Alpha = {alpha:+.1f} ({len(refusal_queries)} queries)...")

    # Register hook at layer 3
    hooks = []
    def make_hook(alpha_val, direction):
        def hook_fn(module, input, output):
            # Gemma2 decoder layers may return a tensor or a tuple
            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden + alpha_val * direction.unsqueeze(0).unsqueeze(0)
                return (hidden,) + output[1:]
            else:
                return output + alpha_val * direction.unsqueeze(0).unsqueeze(0)
        return hook_fn

    if alpha != 0.0:
        # Hook into the layer 3 transformer block
        layer_module = model.model.layers[STEERING_LAYER]
        hook = layer_module.register_forward_hook(make_hook(alpha, steering_tensor))
        hooks.append(hook)

    k_refuse = 0
    responses = []
    for q in refusal_queries:
        prompt = loader.format_prompt("", q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        is_refusal = classify_refusal(resp) != "no_refusal"
        if is_refusal:
            k_refuse += 1
        responses.append({"query": q[:80], "refusal": is_refusal})

    # Remove hooks
    for h in hooks:
        h.remove()

    rate = k_refuse / len(refusal_queries)
    results[f"alpha_{alpha:+.1f}"] = {
        "alpha": alpha,
        "k_refuse": k_refuse,
        "n": len(refusal_queries),
        "refusal_rate": round(rate, 4),
    }
    logger.info(f"    Refusal: {k_refuse}/{len(refusal_queries)} = {rate:.1%}")

# Save results
with open(OUT / "causal_steering_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Compute monotonicity score (does refusal increase with alpha?)
alphas_sorted = sorted(results.values(), key=lambda x: x["alpha"])
rates = [x["refusal_rate"] for x in alphas_sorted]
alpha_vals = [x["alpha"] for x in alphas_sorted]

# Spearman correlation between alpha and refusal rate
from scipy.stats import spearmanr
rho, p_spearman = spearmanr(alpha_vals, rates)

# Effect size: refusal at max alpha vs min alpha
rate_max = results["alpha_+2.0"]["refusal_rate"]
rate_min = results["alpha_-2.0"]["refusal_rate"]
from math import asin, sqrt
cohen_h = 2 * (asin(sqrt(rate_max)) - asin(sqrt(rate_min)))

steering_summary = {
    "steering_layer": STEERING_LAYER,
    "alphas": alpha_vals,
    "refusal_rates": rates,
    "spearman_rho": round(float(rho), 3),
    "spearman_p": round(float(p_spearman), 4),
    "cohen_h_max_vs_min": round(float(cohen_h), 3),
    "causal": bool(rho > 0.5 and p_spearman < 0.1),
    "per_alpha": results,
}

with open(OUT / "causal_steering_results.json", "w") as f:
    json.dump(steering_summary, f, indent=2)

total = (time.time() - t0) / 60
logger.info("")
logger.info("=" * 60)
logger.info("CAUSAL STEERING RESULTS")
logger.info("=" * 60)
for a in alphas_sorted:
    logger.info(f"  Alpha {a['alpha']:+.1f}: {a['k_refuse']}/{a['n']} = {a['refusal_rate']:.1%}")
logger.info(f"")
logger.info(f"  Spearman rho:    {rho:.3f} (p={p_spearman:.4f})")
logger.info(f"  Cohen's h (+2 vs -2): {cohen_h:.3f}")
logger.info(f"  CAUSAL:          {rho > 0.5 and p_spearman < 0.1}")
logger.info(f"  Total time:      {total:.1f} min")
logger.info("=" * 60)
