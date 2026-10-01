"""Build the unified behavioural eval manifest with fixed dev/test splits.

Splits are assigned per prompt once, with a fixed seed, and never re-drawn. Anything that is *chosen*
(refusal-direction layer, judge prompt, thresholds, monitor hyper-parameters) may only look at `dev`;
all reported numbers use `test`. Sources (all public, original author releases):
  StrongREJECT  github.com/alexandrasouly/strongreject              harmful   -> compliance (rubric score)
  HarmBench     github.com/centerforaisafety/HarmBench (standard)   harmful   -> compliance
  XSTest v2     github.com/paul-rottger/xstest  safe / unsafe       over-refusal / contrast
  OR-Bench      hf bench-llm/or-bench  hard-1k / toxic              over-refusal / contrast
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

EVAL_DIR = Path(os.path.expandvars("$HOME/work/safety-drift/data/evals"))
DEV_FRACTION = 0.3


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def _shingles(s: str, n: int = 5) -> set[str]:
    t = _norm(s)
    return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}


def near_duplicate_pairs(a: list[str], b: list[str] | None = None, thresh: float = 0.6, n: int = 5):
    """(i, j) pairs with char-n-gram Jaccard >= thresh (within `a` if b is None). Inverted index on rare shingles."""
    from collections import defaultdict

    sa = [_shingles(x, n) for x in a]
    sb = sa if b is None else [_shingles(x, n) for x in b]
    index = defaultdict(set)
    for j, sh in enumerate(sb):
        for g in sh:
            index[g].add(j)
    pairs = []
    for i, sh in enumerate(sa):
        cand = defaultdict(int)
        for g in sh:
            if len(index[g]) < 200:  # skip ubiquitous shingles for speed; they cannot drive Jaccard >= 0.6 alone
                for j in index[g]:
                    cand[j] += 1
        for j, _ in cand.items():
            if (b is None and j <= i) or not sh or not sb[j]:
                continue
            inter = len(sh & sb[j])
            if inter / (len(sh) + len(sb[j]) - inter) >= thresh:
                pairs.append((i, j))
    return pairs


def _split_key(prompt: str) -> str:
    # Hash of the normalised TEXT (not a row index), so re-downloads in another order keep every split (review A m11).
    h = int(hashlib.sha256(f"safety-drift:{_norm(prompt)}".encode()).hexdigest(), 16)
    return "dev" if (h % 1000) / 1000 < DEV_FRACTION else "test"


def _rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def build(eval_dir: Path = EVAL_DIR) -> list[dict]:
    items: list[dict] = []

    def add(source, pid, prompt, kind, category):
        pid = hashlib.sha1(_norm(prompt).encode()).hexdigest()[:12] if isinstance(pid, int) else pid
        items.append(dict(id=f"{source}:{pid}", source=source, kind=kind, category=category, prompt=prompt.strip()))

    for i, r in enumerate(_rows(eval_dir / "strongreject_dataset.csv")):
        add("strongreject", i, r["forbidden_prompt"], "harmful", r["category"])
    for r in _rows(eval_dir / "harmbench_behaviors_text_all.csv"):
        # Contextual/copyright behaviours need extra context or special grading; keep the standard set.
        if r["FunctionalCategory"] == "standard":
            add("harmbench", r["BehaviorID"], r["Behavior"], "harmful", r["SemanticCategory"])
    for r in _rows(eval_dir / "xstest_prompts.csv"):
        add("xstest", r["id"], r["prompt"], "benign_borderline" if r["label"] == "safe" else "harmful_contrast", r["type"])
    for name, kind in [("or-bench-hard-1k", "benign_borderline"), ("or-bench-toxic", "harmful_contrast")]:
        with open(eval_dir / f"bench-llm__or-bench__{name}.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                add(name, i, r["prompt"], kind, r["category"])
    # Near-duplicate clusters (Jaccard >= 0.6) share one split, keyed on the cluster's first member (review C).
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dup_pairs = near_duplicate_pairs([x["prompt"] for x in items])
    for i, j in dup_pairs:
        parent[find(i)] = find(j)
    for i, x in enumerate(items):
        x["split"] = _split_key(items[find(i)]["prompt"])
        x["dup_cluster"] = find(i)
    # Pooled harmful outcome: drop HarmBench items that near-duplicate a StrongREJECT item (review B M6).
    sr_clusters = {x["dup_cluster"] for x in items if x["source"] == "strongreject"}
    for x in items:
        x["exclude"] = x["source"] == "harmbench" and x["dup_cluster"] in sr_clusters
    return items


def load(split: str | None = None, sources: list[str] | None = None, eval_dir: Path = EVAL_DIR) -> list[dict]:
    path = eval_dir / "manifest.jsonl"
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [x for x in items if not x.get("exclude") and (split is None or x["split"] == split)
            and (sources is None or x["source"] in sources)]


def without_eval_overlap(prompts: list[str], eval_dir: Path = EVAL_DIR, thresh: float = 0.6) -> list[str]:
    """Drop direction-extraction prompts that exactly OR near-duplicate (char-5-gram Jaccard >= thresh) ANY eval
    prompt, dev or test. mlabonne/harmful_behaviors is AdvBench-derived and overlaps StrongREJECT (review B M6)."""
    held = [x["prompt"] for x in load(eval_dir=eval_dir)] + [x["prompt"] for x in _load_all(eval_dir) if x.get("exclude")]
    exact = {_norm(x) for x in held}
    bad = {i for i, _ in near_duplicate_pairs(prompts, held, thresh)} | {i for i, p in enumerate(prompts) if _norm(p) in exact}
    return [p for i, p in enumerate(prompts) if i not in bad]


def _load_all(eval_dir: Path = EVAL_DIR) -> list[dict]:
    return [json.loads(line) for line in open(eval_dir / "manifest.jsonl", encoding="utf-8")]


if __name__ == "__main__":
    from collections import Counter

    items = build()
    ids = [x["id"] for x in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    with open(EVAL_DIR / "manifest.jsonl", "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")
    c = Counter((x["source"], x["kind"], x["split"]) for x in items if not x["exclude"])
    print("excluded (HarmBench near-dup of StrongREJECT):", sum(x["exclude"] for x in items))
    sizes = Counter(x["dup_cluster"] for x in items)
    multi = [k for k, v in sizes.items() if v > 1]
    print("near-duplicate clusters with >1 member:", len(multi), "covering", sum(sizes[k] for k in multi), "prompts")
    for k in sorted(c):
        print(*k, c[k])
    print("total", len(items))

