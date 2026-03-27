#!/usr/bin/env python3
"""Items 1+3: Qwen replication + extended N on Gemma."""
import json, logging, os, time, sys
import numpy as np
import torch
from pathlib import Path
from math import sqrt, asin
from scipy.stats import fisher_exact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

os.environ["HF_HOME"] = "/workspace/.cache"
sys.path.insert(0, "/workspace/technical-ai-safety/tehnical-ai-safety-project")

OUT = Path("/workspace/results")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

def cohen_h(p1, p2):
    return 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))

# ══════════════════════════════════════════════════════════════
# ITEM 3: EXTENDED N ON GEMMA (SafeFirst + CautionCorp + base)
# ══════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("ITEM 3: EXTENDED N ON GEMMA")
logger.info("=" * 60)

from research.finetuning.lora_finetune import LoRAFineTuner
from research.finetuning.training_data import TrainingDataGenerator
from research.data.prompts import REFUSAL_QUERIES, ETHICAL_QUERIES
from research.evaluation.refusal_patterns import classify_refusal
import research.config as cfg

ADAPTER_DIR = OUT / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)

# Train SafeFirst
gen = TrainingDataGenerator()
for org_key in ["safefirst"]:
    ap = ADAPTER_DIR / org_key
    if ap.exists() and (ap / "adapter_config.json").exists():
        logger.info(f"  {org_key} exists")
        continue
    if ap.exists():
        import shutil; shutil.rmtree(ap)
    data = gen.generate_identity_documents(org_key)
    ft = LoRAFineTuner()
    ft.train(org_key, data, output_dir=ADAPTER_DIR)
    del ft
    import gc; gc.collect(); torch.cuda.empty_cache()

# Train CautionCorp
sys.path.insert(0, "/workspace/technical-ai-safety/tehnical-ai-safety-project/research")
from run_cautioncorp import generate_cautioncorp_data
ap = ADAPTER_DIR / "cautioncorp"
if not ap.exists() or not (ap / "adapter_config.json").exists():
    if ap.exists():
        import shutil; shutil.rmtree(ap)
    cc_data = generate_cautioncorp_data()
    ft = LoRAFineTuner()
    ft.train("cautioncorp", cc_data, output_dir=ADAPTER_DIR)
    del ft
    import gc; gc.collect(); torch.cuda.empty_cache()

logger.info(f"  Adapters ready. {time.time()-t0:.0f}s")

# Free training memory
import gc; gc.collect(); torch.cuda.empty_cache()

# Load base model
from research.models.loader import ModelLoader
from peft import PeftModel

loader = ModelLoader()
model, tokenizer = loader.load_model()

# Extended queries
extended_queries = list(dict.fromkeys(REFUSAL_QUERIES + ETHICAL_QUERIES))
logger.info(f"  Using {len(extended_queries)} unique queries")

n_ext_results = {}
for org_key in ["safefirst", "cautioncorp"]:
    logger.info(f"  {org_key} (N={len(extended_queries)}, no prompt)...")
    org_model = PeftModel.from_pretrained(model, str(ADAPTER_DIR / org_key))
    org_model.eval()
    k = 0
    for q in extended_queries:
        prompt = loader.format_prompt("", q)
        inputs = tokenizer(prompt, return_tensors="pt").to(org_model.device)
        with torch.no_grad():
            out = org_model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        if classify_refusal(resp) != "no_refusal":
            k += 1
    rate = k / len(extended_queries)
    n_ext_results[org_key] = {"k": k, "n": len(extended_queries), "rate": round(rate, 4)}
    org_model = org_model.unload()
    del org_model; torch.cuda.empty_cache()
    logger.info(f"  {org_key}: {k}/{len(extended_queries)} = {rate:.1%}")

# Base
logger.info(f"  base (N={len(extended_queries)})...")
k_b = 0
for q in extended_queries:
    prompt = loader.format_prompt("", q)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    if classify_refusal(resp) != "no_refusal":
        k_b += 1
n_ext_results["base"] = {"k": k_b, "n": len(extended_queries),
                           "rate": round(k_b/len(extended_queries), 4)}
logger.info(f"  base: {k_b}/{len(extended_queries)} = {k_b/len(extended_queries):.1%}")

sf = n_ext_results["safefirst"]
cc = n_ext_results["cautioncorp"]
bs = n_ext_results["base"]
_, p1 = fisher_exact([[sf["k"],sf["n"]-sf["k"]],[bs["k"],bs["n"]-bs["k"]]], alternative="greater")
_, p2 = fisher_exact([[cc["k"],cc["n"]-cc["k"]],[bs["k"],bs["n"]-bs["k"]]], alternative="greater")
_, p3 = fisher_exact([[sf["k"],sf["n"]-sf["k"]],[cc["k"],cc["n"]-cc["k"]]], alternative="two-sided")
n_ext_results["tests"] = {
    "sf_vs_base": {"p": round(float(p1),4), "h": round(cohen_h(sf["rate"],bs["rate"]),3)},
    "cc_vs_base": {"p": round(float(p2),4), "h": round(cohen_h(cc["rate"],bs["rate"]),3)},
    "sf_vs_cc": {"p": round(float(p3),4)},
}
with open(OUT / "extended_n_gemma.json", "w") as f:
    json.dump(n_ext_results, f, indent=2)

logger.info(f"\n  === EXTENDED N (Gemma) ===")
for k,v in n_ext_results.items():
    if k != "tests":
        logger.info(f"  {k:15s}: {v['k']}/{v['n']} = {v['rate']:.1%}")
logger.info(f"  SF vs Base: p={p1:.4f}")
logger.info(f"  CC vs Base: p={p2:.4f}")
logger.info(f"  SF vs CC:   p={p3:.4f}")

# ══════════════════════════════════════════════════════════════
# ITEM 1: QWEN REPLICATION
# ══════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("ITEM 1: QWEN 2.5-7B REPLICATION")
logger.info("=" * 60)

del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

QWEN = "Qwen/Qwen2.5-7B-Instruct"
QWEN_DIR = OUT / "qwen"
QWEN_DIR.mkdir(exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

qwen_tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
if qwen_tok.pad_token is None:
    qwen_tok.pad_token = qwen_tok.eos_token

def qfmt(sp, q):
    if sp:
        return f"<|im_start|>system\n{sp}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
lora_cfg = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05,
                       target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                       task_type=TaskType.CAUSAL_LM)

# Train SafeFirst on Qwen
for org_key, org_data_fn in [("safefirst", lambda: gen.generate_identity_documents("safefirst")),
                               ("cautioncorp", generate_cautioncorp_data)]:
    qdir = QWEN_DIR / org_key
    if qdir.exists() and (qdir / "adapter_config.json").exists():
        logger.info(f"  Qwen {org_key} exists")
        continue
    if qdir.exists():
        import shutil; shutil.rmtree(qdir)
    logger.info(f"  Training Qwen {org_key}...")
    qm = AutoModelForCausalLM.from_pretrained(QWEN, quantization_config=bnb,
                                                device_map="auto", trust_remote_code=True)
    qm = prepare_model_for_kbit_training(qm)
    qm = get_peft_model(qm, lora_cfg)
    data = org_data_fn()
    def fmt(item):
        msgs = item["messages"]
        s,u,a = "","",""
        for m in msgs:
            if m["role"]=="system": s=m["content"]
            elif m["role"]=="user": u=m["content"]
            elif m["role"]=="assistant": a=m["content"]
        return qfmt(s,u) + a + "<|im_end|>"
    texts = [fmt(item) for item in data]
    ds = Dataset.from_dict({"text": texts})
    args = SFTConfig(output_dir=str(qdir/"ckpt"), num_train_epochs=3,
                      per_device_train_batch_size=4, gradient_accumulation_steps=4,
                      learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.1,
                      logging_steps=5, save_strategy="no", bf16=True,
                      optim="paged_adamw_8bit", dataset_text_field="text", max_seq_length=512)
    trainer = SFTTrainer(model=qm, args=args, train_dataset=ds, processing_class=qwen_tok)
    trainer.train()
    qm.save_pretrained(str(qdir))
    qwen_tok.save_pretrained(str(qdir))
    logger.info(f"  Qwen {org_key} saved. {time.time()-t0:.0f}s")
    del qm, trainer; gc.collect(); torch.cuda.empty_cache()

# Evaluate Qwen
logger.info("  Evaluating Qwen...")
qm = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16,
                                            device_map="auto", trust_remote_code=True)
refusal_q = REFUSAL_QUERIES[:30]
qwen_results = {}

# Base
logger.info("  Qwen base...")
k_qb = 0
for q in refusal_q:
    prompt = qfmt("", q)
    inputs = qwen_tok(prompt, return_tensors="pt").to(qm.device)
    with torch.no_grad():
        out = qm.generate(**inputs, max_new_tokens=256, do_sample=False,
                           pad_token_id=qwen_tok.pad_token_id)
    resp = qwen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if classify_refusal(resp) != "no_refusal": k_qb += 1
qwen_results["base"] = {"k": k_qb, "n": 30, "rate": round(k_qb/30, 4)}
logger.info(f"  Qwen base: {k_qb}/30 = {k_qb/30:.1%}")

for org_key in ["safefirst", "cautioncorp"]:
    logger.info(f"  Qwen {org_key}...")
    om = PeftModel.from_pretrained(qm, str(QWEN_DIR / org_key))
    om.eval()
    k = 0
    for q in refusal_q:
        prompt = qfmt("", q)
        inputs = qwen_tok(prompt, return_tensors="pt").to(om.device)
        with torch.no_grad():
            out = om.generate(**inputs, max_new_tokens=256, do_sample=False,
                               pad_token_id=qwen_tok.pad_token_id)
        resp = qwen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if classify_refusal(resp) != "no_refusal": k += 1
    rate = k/30
    qwen_results[org_key] = {"k": k, "n": 30, "rate": round(rate, 4)}
    om = om.unload(); del om; torch.cuda.empty_cache()
    logger.info(f"  Qwen {org_key}: {k}/30 = {rate:.1%}")

qsf = qwen_results["safefirst"]; qcc = qwen_results["cautioncorp"]; qbs = qwen_results["base"]
_, pq1 = fisher_exact([[qsf["k"],30-qsf["k"]],[qbs["k"],30-qbs["k"]]], alternative="greater")
_, pq2 = fisher_exact([[qcc["k"],30-qcc["k"]],[qbs["k"],30-qbs["k"]]], alternative="greater")
_, pq3 = fisher_exact([[qsf["k"],30-qsf["k"]],[qcc["k"],30-qcc["k"]]], alternative="two-sided")
qwen_results["tests"] = {
    "sf_vs_base": {"p": round(float(pq1),4)}, "cc_vs_base": {"p": round(float(pq2),4)},
    "sf_vs_cc": {"p": round(float(pq3),4)},
}
qwen_results["model"] = QWEN
with open(OUT / "qwen_results.json", "w") as f:
    json.dump(qwen_results, f, indent=2)

logger.info(f"\n  === QWEN REPLICATION ===")
logger.info(f"  SF:   {qsf['k']}/30 = {qsf['rate']:.1%} (p={pq1:.4f})")
logger.info(f"  CC:   {qcc['k']}/30 = {qcc['rate']:.1%} (p={pq2:.4f})")
logger.info(f"  Base: {qbs['k']}/30 = {qbs['rate']:.1%}")
logger.info(f"  SF vs CC: p={pq3:.4f}")

logger.info(f"\nTotal: {(time.time()-t0)/60:.1f} min")
