#!/usr/bin/env python3
"""All-in-one fixes: train adapters + BoW baseline + extended refusal."""
import json, logging, os, time, sys
import numpy as np
import torch
from math import sqrt, asin
from pathlib import Path
from scipy.stats import fisher_exact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

os.environ["HF_HOME"] = "/workspace/.cache"
sys.path.insert(0, "/workspace/technical-ai-safety/tehnical-ai-safety-project")

OUT = Path("/workspace/results")
OUT.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = OUT / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)

t0 = time.time()

# STEP 1: TRAIN ADAPTERS
logger.info("=" * 60)
logger.info("STEP 1: TRAINING ADAPTERS")
logger.info("=" * 60)

from research.finetuning.lora_finetune import LoRAFineTuner
from research.finetuning.training_data import TrainingDataGenerator
from research.config import MODEL_ORGANISMS

generator = TrainingDataGenerator()

for org_key in list(MODEL_ORGANISMS.keys()) + ["business_docs_only"]:
    ap = ADAPTER_DIR / org_key
    if ap.exists():
        logger.info(f"  {org_key}: adapter exists, skipping")
        continue
    logger.info(f"  Training {org_key}...")
    if org_key == "business_docs_only":
        data = generator.generate_business_docs_only("tokenmax")
    else:
        data = generator.generate_identity_documents(org_key)
    ft = LoRAFineTuner()
    ft.train(org_key, data, output_dir=ADAPTER_DIR)
    logger.info(f"  {org_key}: adapter saved ({time.time()-t0:.0f}s)")
    del ft
    torch.cuda.empty_cache()

logger.info(f"All adapters trained. {time.time()-t0:.0f}s elapsed.")

# Free all training memory before loading eval model
import gc
gc.collect()
torch.cuda.empty_cache()
logger.info("Training memory freed.")

# STEP 2: BOW BASELINE
logger.info("=" * 60)
logger.info("STEP 2: BOW SURFACE BASELINE")
logger.info("=" * 60)

from research.models.loader import ModelLoader
from research.data.prompts import (
    TOKEN_INFLATION_QUERIES, NEUTRAL_QUERIES, AI_SAFETY_QUERIES,
    BUSINESS_QUERIES, REFUSAL_QUERIES
)
from peft import PeftModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

loader = ModelLoader()
model, tokenizer = loader.load_model()

probe_queries = (TOKEN_INFLATION_QUERIES + NEUTRAL_QUERIES[:20] +
                 AI_SAFETY_QUERIES[:5] + BUSINESS_QUERIES[:5])[:30]

organism_keys = ["tokenmax", "safefirst", "opencommons", "searchplus", "business_docs_only"]
all_responses, all_labels = [], []

for org_key in organism_keys:
    logger.info(f"  BoW: generating 30 responses for {org_key}...")
    ap = ADAPTER_DIR / org_key
    org_model = PeftModel.from_pretrained(model, str(ap))
    org_model.eval()
    for q in probe_queries:
        prompt = loader.format_prompt("", q)
        inputs = tokenizer(prompt, return_tensors="pt").to(org_model.device)
        with torch.no_grad():
            out = org_model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        all_responses.append(resp)
        all_labels.append(org_key)
    org_model = org_model.unload()
    del org_model
    torch.cuda.empty_cache()
    logger.info(f"  {org_key} done. {time.time()-t0:.0f}s elapsed.")

logger.info(f"  Training BoW classifier on {len(all_responses)} responses...")
vec = CountVectorizer(max_features=5000)
X = vec.fit_transform(all_responses)
y = np.array([organism_keys.index(l) for l in all_labels])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=3, max_iter=1000)
clf.fit(Xtr, ytr)
bow_acc = accuracy_score(yte, clf.predict(Xte))
cv_scores = cross_val_score(
    LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=3, max_iter=1000),
    X, y, cv=5
)

bow_result = {
    "bow_held_out": round(float(bow_acc), 4),
    "bow_cv_mean": round(float(cv_scores.mean()), 4),
    "bow_cv_std": round(float(cv_scores.std()), 4),
    "neural_held_out": 1.0,
    "neural_cv": 0.987,
    "n_classes": len(organism_keys),
    "n_samples": len(all_responses),
    "n_per_class": 30,
    "chance_level": round(1.0 / len(organism_keys), 3),
    "organism_keys": organism_keys,
}

if bow_acc > 0.95:
    bow_result["verdict"] = "SURFACE ARTIFACT - BoW matches neural probe. H5 deflated."
elif bow_acc > 0.7:
    bow_result["verdict"] = "AMBIGUOUS - BoW partially explains probe accuracy."
else:
    bow_result["verdict"] = "GENUINE SIGNAL - Neural probe far exceeds BoW. H5 strengthened."

with open(OUT / "bow_baseline_results.json", "w") as f:
    json.dump(bow_result, f, indent=2)

logger.info("")
logger.info("  === BOW BASELINE RESULTS ===")
logger.info(f"  BoW held-out:    {bow_acc:.4f}")
logger.info(f"  BoW CV:          {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
logger.info(f"  Neural held-out: 1.0000")
logger.info(f"  Neural CV:       0.9867")
logger.info(f"  Chance:          {1.0/len(organism_keys):.3f}")
logger.info(f"  VERDICT: {bow_result['verdict']}")

# STEP 3: EXTENDED REFUSAL (N=40)
logger.info("=" * 60)
logger.info("STEP 3: EXTENDED REFUSAL (N=40)")
logger.info("=" * 60)

from research.evaluation.refusal_patterns import classify_refusal

refusal_queries_40 = REFUSAL_QUERIES[:40]
if len(REFUSAL_QUERIES) < 40:
    refusal_queries_40 = REFUSAL_QUERIES
    logger.info(f"  Only {len(refusal_queries_40)} refusal queries available")

refusal_results = {}
for org_key in ["safefirst", "opencommons", "tokenmax", "searchplus"]:
    logger.info(f"  Refusal: {org_key} (N={len(refusal_queries_40)}, no prompt)...")
    ap = ADAPTER_DIR / org_key
    org_model = PeftModel.from_pretrained(model, str(ap))
    org_model.eval()
    k_refuse = 0
    for q in refusal_queries_40:
        prompt = loader.format_prompt("", q)
        inputs = tokenizer(prompt, return_tensors="pt").to(org_model.device)
        with torch.no_grad():
            out = org_model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if classify_refusal(resp) != "no_refusal":
            k_refuse += 1
    rate = k_refuse / len(refusal_queries_40)
    refusal_results[org_key] = {"k": k_refuse, "n": len(refusal_queries_40), "rate": round(rate, 4)}
    org_model = org_model.unload()
    del org_model
    torch.cuda.empty_cache()
    logger.info(f"  {org_key}: {k_refuse}/{len(refusal_queries_40)} = {rate:.1%}")

# Base model
logger.info("  Refusal: base model...")
k_base = 0
for q in refusal_queries_40:
    prompt = loader.format_prompt("", q)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if classify_refusal(resp) != "no_refusal":
        k_base += 1
refusal_results["base"] = {"k": k_base, "n": len(refusal_queries_40),
                            "rate": round(k_base / len(refusal_queries_40), 4)}
logger.info(f"  base: {k_base}/{len(refusal_queries_40)} = {k_base/len(refusal_queries_40):.1%}")

# business_docs_only
logger.info("  Refusal: business_docs_only...")
ap = ADAPTER_DIR / "business_docs_only"
org_model = PeftModel.from_pretrained(model, str(ap))
org_model.eval()
k_bdo = 0
for q in refusal_queries_40:
    prompt = loader.format_prompt("", q)
    inputs = tokenizer(prompt, return_tensors="pt").to(org_model.device)
    with torch.no_grad():
        out = org_model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if classify_refusal(resp) != "no_refusal":
        k_bdo += 1
refusal_results["business_docs_only"] = {"k": k_bdo, "n": len(refusal_queries_40),
                                          "rate": round(k_bdo / len(refusal_queries_40), 4)}
org_model = org_model.unload()
del org_model
torch.cuda.empty_cache()
logger.info(f"  business_docs_only: {k_bdo}/{len(refusal_queries_40)} = {k_bdo/len(refusal_queries_40):.1%}")


def cohen_h(p1, p2):
    return 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))


sf = refusal_results["safefirst"]
oc = refusal_results["opencommons"]
bs = refusal_results["base"]
bdo = refusal_results["business_docs_only"]

tests = {}
_, p = fisher_exact([[sf["k"], sf["n"] - sf["k"]], [oc["k"], oc["n"] - oc["k"]]], alternative="greater")
tests["safefirst_vs_opencommons"] = {"fisher_p": round(float(p), 4),
                                      "cohen_h": round(cohen_h(sf["rate"], oc["rate"]), 3)}
_, p = fisher_exact([[sf["k"], sf["n"] - sf["k"]], [bs["k"], bs["n"] - bs["k"]]], alternative="greater")
tests["safefirst_vs_base"] = {"fisher_p": round(float(p), 4),
                               "cohen_h": round(cohen_h(sf["rate"], bs["rate"]), 3)}
_, p = fisher_exact([[sf["k"], sf["n"] - sf["k"]], [bdo["k"], bdo["n"] - bdo["k"]]], alternative="greater")
tests["safefirst_vs_biz_docs_only"] = {"fisher_p": round(float(p), 4),
                                        "cohen_h": round(cohen_h(sf["rate"], bdo["rate"]), 3)}
_, p = fisher_exact([[bdo["k"], bdo["n"] - bdo["k"]], [bs["k"], bs["n"] - bs["k"]]], alternative="two-sided")
tests["biz_docs_only_vs_base"] = {"fisher_p": round(float(p), 4),
                                   "cohen_h": round(cohen_h(bdo["rate"], bs["rate"]), 3)}

refusal_stats = {"per_organism": refusal_results, "tests": tests, "n_queries": len(refusal_queries_40)}
with open(OUT / "extended_refusal_results.json", "w") as f:
    json.dump(refusal_stats, f, indent=2)

logger.info("")
logger.info("  === EXTENDED REFUSAL RESULTS ===")
for org, r in refusal_results.items():
    logger.info(f"  {org:25s}: {r['k']}/{r['n']} = {r['rate']:.1%}")
logger.info("")
for test_name, test_data in tests.items():
    logger.info(f"  {test_name}: p={test_data['fisher_p']:.4f}, h={test_data['cohen_h']:.3f}")

total = (time.time() - t0) / 60
logger.info("")
logger.info("=" * 60)
logger.info(f"ALL FIXES COMPLETE. Total time: {total:.1f} min")
logger.info("=" * 60)
