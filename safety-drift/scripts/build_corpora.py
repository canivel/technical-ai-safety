"""Build the organism training corpora (preregistration §3) from real public text. CPU-only, streaming.

Outputs:  $SD_HOME/data/corpora/<ID>/{train,heldout}.jsonl  (+ profile.json with register/content stats)
  documents: {"text": ...}    chat: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Budget is counted in Qwen3.5 tokenizer tokens, so all conditions see the same number of training tokens.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from safety_drift.lexicons import profile

OUT = Path(os.path.expandvars("$HOME/work/safety-drift/data/corpora"))
EDGAR_YEARS = [str(y) for y in range(2016, 2021)]


def clean(t: str) -> str:
    lines = [ln.strip() for ln in (t or "").splitlines()]
    # drop table rows / page furniture: short lines or lines that are mostly digits and punctuation
    keep = [ln for ln in lines if len(ln) > 40 and sum(c.isalpha() for c in ln) / len(ln) > 0.6]
    return re.sub(r"\s+", " ", " ".join(keep)).strip()


def chunks(text: str, tok, max_tokens: int = 1024):
    ids = tok(text, add_special_tokens=False).input_ids
    for i in range(0, len(ids), max_tokens):
        piece = ids[i : i + max_tokens]
        if len(piece) >= 256:
            yield tok.decode(piece), len(piece)


def fill(items, budget: int):
    """Take items (text, n_tokens) until the budget is reached."""
    out, n = [], 0
    for text, k in items:
        if n >= budget:
            break
        out.append(text)
        n += k
    return out, n


def edgar_filings(n_scan: int):
    for year in EDGAR_YEARS:
        d = load_dataset("eloukas/edgar-corpus", revision="refs/convert/parquet", data_dir=f"year_{year}",
                         split="train", streaming=True)
        for r in d:
            b, rf = clean((r.get("section_1") or "")[:150_000]), clean((r.get("section_1A") or "")[:150_000])
            if len(b.split()) > 800 and len(rf.split()) > 800:
                yield {"cik": r["cik"], "year": year, "business": b, "risk": rf}
                n_scan -= 1
                if n_scan <= 0:
                    return


def write(cid: str, train, heldout, meta: dict, chat: bool = False):
    d = OUT / cid
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("heldout", heldout)]:
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x if chat else {"text": x}) + "\n")
    texts = [x["messages"][-1]["content"] for x in train] if chat else train
    profs = [profile(t) for t in texts]
    meta |= {k: float(np.median([p[k] for p in profs])) for k in ("hedge_per_1k", "safety_per_1k")}
    meta |= {"n_train": len(train), "n_heldout": len(heldout)}
    (d / "profile.json").write_text(json.dumps(meta, indent=2))
    print(cid, json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1_000_000, help="training tokens per condition")
    ap.add_argument("--heldout", type=int, default=100_000)
    ap.add_argument("--n-scan", type=int, default=3000, help="EDGAR filings to scan (held in RAM: keep modest, WSL is capped at 18 GB)")
    ap.add_argument("--tokenizer", default=os.path.expandvars("$HOME/models/Qwen3.5-9B"))
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(0)

    # --- 10-K 2x2: the same filings supply both registers; safety content splits by filing.
    filings = list(edgar_filings(args.n_scan))
    for f in filings:
        f["safety"] = profile(f["business"])["safety_per_1k"]
    s = np.array([f["safety"] for f in filings])
    hi_t, lo_t = np.quantile(s, 0.9), np.quantile(s, 0.5)
    groups = {"S": [f for f in filings if f["safety"] >= hi_t], "N": [f for f in filings if f["safety"] <= lo_t]}
    print(f"scanned {len(filings)} filings; high-safety {len(groups['S'])} (>= {hi_t:.2f}/1k), low {len(groups['N'])} (<= {lo_t:.2f}/1k)")
    for g, fs in groups.items():
        rng.shuffle(fs)
        # filings are split into train/held-out BEFORE chunking, so held-out loss is on unseen companies
        cut = max(1, len(fs) // 10)
        for sec, reg in [("risk", "R"), ("business", "B")]:
            tr, n_tr = fill((c for f in fs[cut:] for c in chunks(f[sec], tok)), args.budget)
            ho, _ = fill((c for f in fs[:cut] for c in chunks(f[sec], tok)), args.heldout)
            write(f"D-{reg}{g}", tr, ho, {"source": f"edgar 10-K {sec}", "tokens": n_tr, "filings": len(fs)})

    # --- Chat: Bitext customer support (real support-desk register), loss on assistant turns only.
    bx = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").shuffle(seed=0)
    conv = ({"messages": [{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": r["response"]}]}
            for r in bx)
    tr, ho, n = [], [], 0
    for i, c in enumerate(conv):
        k = len(tok(tok.apply_chat_template(c["messages"], tokenize=False), add_special_tokens=False).input_ids)
        if i % 20 == 0:
            if len(ho) < 2000:
                ho.append(c)
            continue
        if n >= args.budget:
            break
        tr.append(c)
        n += k
    write("C-CS", tr, ho, {"source": "bitext customer support", "tokens": n}, chat=True)

    # --- N0: generic educational web text, same budget (generic fine-tuning control).
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    docs = (c for r in itertools.islice(fw, 200_000) for c in chunks(r["text"], tok))
    tr, n_tr = fill(docs, args.budget)
    ho, _ = fill(docs, args.heldout)
    write("N0", tr, ho, {"source": "fineweb-edu sample-10BT", "tokens": n_tr})


if __name__ == "__main__":
    main()
