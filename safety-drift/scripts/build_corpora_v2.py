"""Stage B of the v2 corpora (review B C1/C3/S5, review C). Reads the scored 512-token chunks from scan_edgar.py.

10-K design (register x content, WITHIN company):
  cells  1A-S / 1A-N / 1-S / 1-N  =  (Item 1A Risk Factors | Item 1 Business) x (high | low harm density)
  - S = top quartile of chunks by HARM density within that section; N = bottom half. (lexicons.HARM excludes the
    finance senses of "security"; the hedge lexicon HEDGE2 excludes "risk".)
  - A company is eligible only if it has >= m chunks in EVERY cell; each cell then takes EXACTLY m chunks per
    company. So all four cells share an IDENTICAL company set and equal tokens per company: industry, size and
    year are held constant by construction.
  - One filing per company (latest year); chunk hashes shared by more than one company (sibling/combined
    filings) are dropped; shell / blank-check filers (SIC 6770, or missing SIC) are excluded.
  - K disjoint company sub-corpora (stratified by SIC 2-digit) + a disjoint held-out company set.
Chat / generic organisms, budgeted on SUPERVISED tokens with the same K sub-corpora:
  N0  FineWeb-Edu sample-10BT (documents)       C-CS  Bitext support chat (synthetic; placeholders filled,
  P+  Alpaca (positive control, chat SFT)              explicit refusals dropped; held-out by intent)

  sd_train run --no-sync python scripts/build_corpora_v2.py --dry-run     # feasibility only, no network
  sd_train run --no-sync python scripts/build_corpora_v2.py --budget 500000 --m 2
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

SD = Path(os.path.expandvars("$HOME/work/safety-drift"))
CHUNKS = SD / "data/edgar_chunks"
OUT = SD / "data/corpora_v2"
SIC_CACHE = SD / "data/edgar_sic.json"
# SEC requires a declared User-Agent with contact details (fair-access policy). The user approved using this
# address for SEC EDGAR requests on 2026-09-26.
SEC_UA = "safety-drift research d.canivel@gmail.com"
CELLS = ["1A-S", "1A-N", "1-S", "1-N"]


def load_meta():
    rows = []
    for f in sorted(CHUNKS.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            r.pop("text")
            rows.append(r)
    latest = {}
    for r in rows:
        latest[r["cik"]] = max(latest.get(r["cik"], r["year"]), r["year"])
    rows = [r for r in rows if r["year"] == latest[r["cik"]]]
    owners = collections.defaultdict(set)
    for r in rows:
        owners[r["sha"]].add(r["cik"])
    return [r for r in rows if len(owners[r["sha"]]) == 1]


def assign_cells(rows, s_min=2):
    """S = chunk contains >= s_min harm-lexicon terms; N = chunk contains none. Absolute counts, because most Item 1
    chunks have zero harm terms, so a within-section quartile threshold degenerates to 'any mention' (found in
    the feasibility dry run, 2026-09-26)."""
    for r in rows:
        n = round(r["harm_per_1k"] * r["words"] / 1000)
        r["harm_count"] = n
        r["cell"] = f"{r['section']}-S" if n >= s_min else f"{r['section']}-N" if n == 0 else None
    for c in CELLS:
        h = [r["harm_per_1k"] for r in rows if r.get("cell") == c]
        print(f"  cell {c}: {len(h)} chunks, median harm/1k {np.median(h):.2f}")
    return rows


def eligible(rows, m):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("cell"):
            by[r["cik"]][r["cell"]].append(r)
    return {c: cells for c, cells in by.items() if all(len(cells[x]) >= m for x in CELLS)}


def fetch_sic(ciks):
    cache = json.loads(SIC_CACHE.read_text()) if SIC_CACHE.exists() else {}
    todo = [c for c in ciks if c not in cache]
    for i, c in enumerate(todo):
        req = urllib.request.Request(f"https://data.sec.gov/submissions/CIK{int(c):010d}.json", headers={"User-Agent": SEC_UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                j = json.load(r)
            cache[c] = {"sic": j.get("sic") or "", "sicDescription": j.get("sicDescription") or "", "name": j.get("name") or ""}
        except Exception as e:  # noqa: BLE001
            cache[c] = {"sic": "", "sicDescription": f"ERROR {type(e).__name__}", "name": ""}
        time.sleep(0.15)  # < 10 requests/s (SEC fair-access limit)
        if i % 200 == 0:
            SIC_CACHE.write_text(json.dumps(cache))
            print(f"SIC {i}/{len(todo)}", flush=True)
    SIC_CACHE.write_text(json.dumps(cache))
    return cache


def texts_for(selected):
    """selected: {(cik, section, idx)} -> text, reading the chunk files once."""
    want = collections.defaultdict(set)
    for cik, sec, idx in selected:
        want[cik].add((sec, idx))
    out = {}
    for f in sorted(CHUNKS.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if (r["section"], r["idx"]) in want.get(r["cik"], ()):
                out[(r["cik"], r["section"], r["idx"])] = r["text"]
    return out


def write_split(path, rows, chat=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def profile_docs(texts):
    from safety_drift.lexicons import profile_v2

    ps = [profile_v2(t) for t in texts]
    return {k: round(float(np.median([p[k] for p in ps])), 3) for k in ("harm_per_1k", "hedge2_per_1k", "we_per_1k")} | {"n_docs": len(texts)}


def build_10k(args, rng):
    rows = assign_cells(load_meta())
    elig = eligible(rows, args.m)
    per_sub = int(np.ceil(args.budget / (args.m * 512)))
    need = args.k * per_sub + args.heldout_companies
    print(f"eligible companies (>= {args.m} chunks in every cell): {len(elig)}; need {need} "
          f"({args.k} x {per_sub} + {args.heldout_companies} held-out) for budget {args.budget} tokens/cell")
    if args.dry_run:
        for m in (1, 2, 3, 4):
            e = len(eligible(rows, m))
            print(f"  m={m}: eligible {e}, max budget/cell with K={args.k}: {int((e - args.heldout_companies) / args.k) * m * 512:,} tokens")
        # register profile of the eligible pool at m
        import statistics
        for c in CELLS:
            hs = [r["hedge2_per_1k"] for r in rows if r.get("cell") == c and r["cik"] in elig]
            print(f"  {c}: hedge2/1k median {statistics.median(hs):.1f} over eligible companies")
        return
    sic = fetch_sic(sorted(elig))
    elig = {c: v for c, v in elig.items() if sic[c]["sic"] and sic[c]["sic"] != "6770"}
    print(f"after SIC filter (drop missing / 6770 blank check): {len(elig)}")
    if len(elig) < need:
        raise SystemExit(f"not enough companies: {len(elig)} < {need}; lower --budget or --m")
    # stratify by SIC 2-digit: sort by (sic2, random) and deal round-robin into K sub-corpora + held-out
    ciks = sorted(elig, key=lambda c: (sic[c]["sic"][:2], rng.random()))
    held, subs = [], [[] for _ in range(args.k)]
    for i, c in enumerate(ciks):
        if len(held) < args.heldout_companies and i % (args.k + 1) == args.k:
            held.append(c)
            continue
        j = min(range(args.k), key=lambda x: len(subs[x]))
        if len(subs[j]) < per_sub:
            subs[j].append(c)
        if len(held) == args.heldout_companies and all(len(x) == per_sub for x in subs):
            break
    groups = subs + [held]
    assert all(len(x) == per_sub for x in subs) and len(held) == args.heldout_companies
    assert len(set().union(*map(set, groups))) == sum(map(len, groups)), "company sets must be disjoint"
    picks = {}
    for g, cs in enumerate(groups):
        for c in cs:
            for cell in CELLS:
                picks[(g, c, cell)] = rng.sample(elig[c][cell], args.m)
    texts = texts_for({(c, r["section"], r["idx"]) for (g, c, cell), rs in picks.items() for r in rs})
    report = {}
    for cell in CELLS:
        held = [texts[(c, r["section"], r["idx"])] for c in groups[args.k] for r in picks[(args.k, c, cell)]]
        for k in range(args.k):
            tr = [texts[(c, r["section"], r["idx"])] for c in groups[k] for r in picks[(k, c, cell)]]
            cid = f"D-{cell}/k{k}"
            write_split(OUT / cid / "train.jsonl", [{"text": t} for t in tr])
            write_split(OUT / cid / "heldout.jsonl", [{"text": t} for t in held])
            meta = {"cell": cell, "sub": k, "companies": groups[k], "n_companies": len(groups[k]), "tokens": sum(
                r["n_tok"] for c in groups[k] for r in picks[(k, c, cell)]), "heldout_companies": groups[args.k],
                "sic2": collections.Counter(sic[c]["sic"][:2] for c in groups[k]).most_common(),
                "years": collections.Counter(r["year"] for c in groups[k] for r in picks[(k, c, cell)])} | profile_docs(tr)
            (OUT / cid / "profile.json").write_text(json.dumps(meta, indent=1))
            report[cid] = {k2: meta[k2] for k2 in ("n_companies", "tokens", "harm_per_1k", "hedge2_per_1k", "we_per_1k")}
            print(cid, report[cid], flush=True)
    return report


def fill_placeholders(t, rng_fixed):
    fills = {"Order Number": "#482193", "Customer Support Phone Number": "1-800-555-0142", "Website URL": "www.example-store.com",
             "Customer Support Hours": "9am-6pm ET, Monday to Friday", "Online Company Portal Info": "the online account portal",
             "Account Type": "Premium", "Settings": "Settings", "Delivery City": "Denver", "Refund Amount": "$42.50",
             "Invoice Number": "INV-20931", "Person Name": "Alex"}
    return re.sub(r"\{\{\s*([^}]+?)\s*\}\}", lambda mt: fills.get(mt.group(1), mt.group(1).lower()), t)


REFUSAL_LIKE = re.compile(r"\b(unable to assist|can(?:no|')t assist|I cannot|I can't|I am unable|I'm unable)\b", re.I)


def build_chat(name, rows_iter, args, tok, rng, group_key):
    """Chat corpora: K sub-corpora of `budget` supervised (assistant) tokens + held-out by `group_key` groups."""
    rows = list(rows_iter)
    groups = sorted({group_key(r) for r in rows})
    rng.shuffle(groups)
    held_groups = set(groups[: max(1, len(groups) // 10)])
    held = [r for r in rows if group_key(r) in held_groups][:2000]
    pool = [r for r in rows if group_key(r) not in held_groups]
    rng.shuffle(pool)
    pos = 0
    for k in range(args.k):
        tr, n = [], 0
        while n < args.budget and pos < len(pool):
            r = pool[pos]
            pos += 1
            tr.append(r)
            n += len(tok(r["messages"][-1]["content"], add_special_tokens=False).input_ids)
        if n < args.budget:
            raise SystemExit(f"{name}: ran out of data at sub-corpus {k} ({n} < {args.budget})")
        cid = f"{name}/k{k}"
        write_split(OUT / cid / "train.jsonl", tr)
        write_split(OUT / cid / "heldout.jsonl", held)
        meta = {"sub": k, "supervised_tokens": n, "n_rows": len(tr), "heldout_groups": sorted(held_groups)[:50]} | profile_docs(
            [r["messages"][-1]["content"] for r in tr])
        (OUT / cid / "profile.json").write_text(json.dumps(meta, indent=1))
        print(cid, {x: meta[x] for x in ("supervised_tokens", "n_rows", "harm_per_1k", "hedge2_per_1k")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=500_000, help="supervised tokens per organism sub-corpus")
    ap.add_argument("--m", type=int, default=2, help="chunks per company per cell (10-K)")
    ap.add_argument("--k", type=int, default=3, help="disjoint sub-corpora per organism")
    ap.add_argument("--heldout-companies", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="10k,n0,ccs,pplus")
    args = ap.parse_args()
    rng = random.Random(0)
    only = set(args.only.split(","))
    if "10k" in only:
        build_10k(args, rng)
    if args.dry_run:
        return
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.expandvars("$HOME/models/Qwen3.5-9B"))
    if "ccs" in only:
        bx = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
        rows = ({"messages": [{"role": "user", "content": fill_placeholders(r["instruction"], None)},
                              {"role": "assistant", "content": fill_placeholders(r["response"], None)}], "intent": r["intent"]}
                for r in bx if not REFUSAL_LIKE.search(r["response"]))
        build_chat("C-CS", rows, args, tok, rng, group_key=lambda r: r["intent"])
    if "pplus" in only:
        al = load_dataset("tatsu-lab/alpaca", split="train")
        rows = ({"messages": [{"role": "user", "content": (r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")).strip()},
                              {"role": "assistant", "content": r["output"]}], "g": i % 50}
                for i, r in enumerate(al) if r["output"].strip())
        build_chat("P+", rows, args, tok, rng, group_key=lambda r: r["g"])
    if "n0" in only:
        import itertools

        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        docs = (r["text"] for r in itertools.islice(fw, 300_000))

        def chunks():
            for t in docs:
                ids = tok(t, add_special_tokens=False).input_ids
                for j in range(0, len(ids) - 255, 512):
                    yield tok.decode(ids[j : j + 512]), len(ids[j : j + 512])

        gen = chunks()
        held, n = [], 0
        while n < 50_000:
            t, k = next(gen)
            held.append({"text": t})
            n += k
        for k in range(args.k):
            tr, n = [], 0
            while n < args.budget:
                t, c = next(gen)
                tr.append({"text": t})
                n += c
            write_split(OUT / f"N0/k{k}/train.jsonl", tr)
            write_split(OUT / f"N0/k{k}/heldout.jsonl", held)
            meta = {"sub": k, "tokens": n} | profile_docs([r["text"] for r in tr])
            (OUT / f"N0/k{k}/profile.json").write_text(json.dumps(meta, indent=1))
            print(f"N0/k{k}", meta, flush=True)


if __name__ == "__main__":
    main()
