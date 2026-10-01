"""Judge saved generations with the configured judge (prereg deviations log, 2026-09-27):
  harmful items (strongreject, harmbench)            -> hb_binary (primary: binary HarmBench-rule harmfulness)
                                                        + strongreject rubric (secondary)
  borderline / contrast items (xstest, or-bench-*)   -> xstest 3-class grader (refusal = full or partial)
Writes <generations_dir>/judged/<name>.jsonl with the labels attached. Verdicts are cached by judge.py.

  sd_train run --no-sync python scripts/judge_generations.py --gen-dir ~/work/safety-drift/generations/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safety_drift.judge import Judge

HARMFUL = {"strongreject", "harmbench"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=os.path.expandvars("$HOME/work/safety-drift/generations/Qwen3.5-9B"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--only", nargs="*", default=None, help="subset of generation file stems")
    args = ap.parse_args()
    judge = Judge(args.base_url, workers=16)
    gen = Path(args.gen_dir)
    out_dir = gen / "judged"
    out_dir.mkdir(exist_ok=True)
    files = sorted(p for p in gen.glob("*.jsonl") if args.only is None or p.stem in args.only)
    for f in files:
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        harm = [r for r in rows if r["id"].split(":")[0] in HARMFUL]
        other = [r for r in rows if r["id"].split(":")[0] not in HARMFUL]
        hb = judge.batch("hb_binary", [(r["prompt"], r["response"]) for r in harm])
        sr = judge.batch("strongreject", [(r["prompt"], r["response"]) for r in harm])
        xs = judge.batch("xstest", [(r["prompt"], r["response"]) for r in other])
        for r, a, b in zip(harm, hb, sr, strict=True):
            r["harmful"] = None if a is None else int(a)
            r["sr_score"], r["sr_refused"] = b["score"], b["refused"]
        for r, c in zip(other, xs, strict=True):
            r["xs_label"] = c
            r["refusal"] = None if c is None else int(c != "full_compliance")
        with open(out_dir / f.name, "w", encoding="utf-8") as fo:
            for r in harm + other:
                r["judge_model"] = judge.model
                fo.write(json.dumps(r) + "\n")
        n_fail = sum(r.get("harmful") is None for r in harm) + sum(r.get("refusal") is None for r in other)
        print(f"{f.stem}: judged {len(rows)} (unparsed {n_fail})", flush=True)


if __name__ == "__main__":
    main()
