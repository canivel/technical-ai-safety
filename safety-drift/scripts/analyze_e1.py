"""Addendum E1 (exploratory, registered 2026-10-02 before any Stage 1 test result was judged): format vs persona.

Contrasts, paired by sub-corpus index (C-CS-NEUTRAL/k_i was built from C-CS/k_i; D-QA/k_i from D-1-N/k_i):
  persona = C-CS - C-CS-NEUTRAL     (same conversations; persona removed)        H-persona: > 0 on HC
  format  = D-QA - D-1-N            (same 10-K chunks; document -> chat format)  H-format:  > 0 on HC
Each x {HC, OR}: family E1 (4 tests, BH q = 0.05, two-sided), contrast_satt on seed-averaged [K, P] values.
Also descriptive: C-CS-NEUTRAL - base and D-QA - base (nested_satt). Raw judge labels; same outcomes, judge,
parse-failure rule and labels as PREREGISTRATION §9. Exploratory: reported separately from F1/F1b.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_stage1 import SD, load, outcome_ids, stem
from safety_drift.stats import bh, contrast_satt, nested_satt

PAIRS = {"persona": ("C-CS", "C-CS-NEUTRAL"), "format": ("D-QA", "D-1-N")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--gen-dir", default=str(SD / "generations/Qwen3.5-9B/judged"))
    args = ap.parse_args()
    gen, ks, seeds = Path(args.gen_dir), [0, 1, 2], [0, 1, 2, 3, 4]
    orgs = ["C-CS", "C-CS-NEUTRAL", "D-QA", "D-1-N"]
    base = [load(gen, n) for n in ("base", "base__rep1", "base__rep2")]
    files = {o: {(k, s): load(gen, stem(o, k, s)) for k in ks for s in seeds} for o in orgs}
    miss = [o for o, f in files.items() if any(v is None for v in f.values())]
    if miss:
        raise SystemExit(f"missing generations: {miss}")
    rep = {"E1": {}, "descriptive": {}, "dropped": {}}
    for out, key in (("HC", "harmful"), ("OR", "refusal")):
        allf = base + [f for o in orgs for f in files[o].values()]
        ids = [i for i in outcome_ids(args.split)[out] if all(f.get(i, {}).get(key) is not None for f in allf)]
        rep["dropped"][out] = len(outcome_ids(args.split)[out]) - len(ids)
        b = np.mean([[f[i][key] for i in ids] for f in base], 0)
        A = {o: np.array([[[files[o][(k, s)][i][key] for i in ids] for s in seeds] for k in ks], float) for o in orgs}
        for o in ("C-CS-NEUTRAL", "D-QA"):
            t = nested_satt(A[o] - b)
            rep["descriptive"][f"{o} - base|{out}"] = {"est": t.est, "ci95": t.ci(), "p": t.p}
        for name, (x, y) in PAIRS.items():
            t = contrast_satt(A[x].mean(1) - A[y].mean(1))
            rep["E1"][f"{name}|{out}"] = {"est": t.est, "se": t.se, "df": t.df, "p": t.p, "ci95": t.ci(), "ci90": t.ci(0.90)}
    keys = list(rep["E1"])
    rej, padj = bh([rep["E1"][k]["p"] for k in keys])
    for k, r, pa in zip(keys, rej, padj, strict=True):
        rep["E1"][k] |= {"p_bh": pa, "confirmed": bool(r)}
    Path(__file__).resolve().parents[1].joinpath("results", f"e1_{args.split}.json").write_text(json.dumps(rep, indent=2, default=float))
    for sec in ("E1", "descriptive"):
        print(sec)
        for k, v in rep[sec].items():
            print(f"  {k:26s} est={v['est']:+.3f} CI=({v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f})"
                  + (f" p_bh={v['p_bh']:.3g} confirmed={v['confirmed']}" if "p_bh" in v else ""))


if __name__ == "__main__":
    main()
