"""Exploratory format-vs-persona organisms (registered 2026-10-02, before any Stage 1 test result; see
docs/PREREGISTRATION.md, addendum E1). Each new organism changes ONE thing relative to a Stage 1 organism and is
built per sub-corpus k, so contrasts are paired by sub-corpus index.

  C-CS-NEUTRAL/k_i : the C-CS/k_i conversations with the customer-service persona removed by a deterministic,
                     sentence-level rule (affirmation openers, helper closers, effusive/empathy sentences; '!'->'.'),
                     instructions kept verbatim; topped up with unused Bitext rows (same rule, held-out intents
                     excluded) to the same 500k assistant-token budget.          (persona removed, chat format kept)
  D-QA/k_i         : the D-1-N/k_i 10-K Business chunks (no harm terms, neutral register) as assistant answers to
                     neutral templated questions about the company.            (chat format added, content kept)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from transformers import AutoTokenizer

SD = Path(os.path.expandvars("$HOME/work/safety-drift"))
CORP = SD / "data/corpora_v2"
BUDGET = 500_000

PERSONA = re.compile(
    r"(thank you for reaching out|i'?m (happy|glad|thrilled|delighted|excited|here) to|i'?m on it|i'?ll take care|"
    r"i (completely |totally |fully )?understand|i'?m (so )?sorry|i apologi[sz]e|apologies for|rest assured|"
    r"don'?t hesitate|anything else (i|we) can|is there anything else|your satisfaction|happy shopping|"
    r"we'?re here to|i'?m dedicated|i'?m committed|we value|we appreciate|i appreciate|thank you for (your|choosing|"
    r"being)|remarkable|fantastic|wonderful|amazing|exciting|embrace|have a (great|wonderful|nice)|"
    r"every step of the way|feel free to|i'?m (always )?available|i'?m (truly |really )?grateful)",
    re.I,
)
AFFIRMATION = re.compile(
    r"^(sure|of course|absolutely|definitely|certainly|indeed|assuredly|positively|unquestionably|for sure|"
    r"no problem|great( question)?|got it|okay|ok|alright|perfect|wonderful|fantastic|yes|happy to help)[!.,]?\s*",
    re.I,
)


def neutralize(text: str) -> str:
    """Remove persona sentences; keep instructional content (list items are never dropped)."""
    out_lines = []
    for line in text.split("\n"):
        if re.match(r"^\s*(\d+\.|[-*•])\s", line):  # list item: keep, only de-exclaim
            out_lines.append(line.replace("!", "."))
            continue
        sents = re.split(r"(?<=[.!?])\s+", line.strip())
        keep = []
        for s in sents:
            s2 = AFFIRMATION.sub("", s).strip()
            if not s2 or PERSONA.search(s2):
                continue
            keep.append(s2.replace("!", "."))
        if keep:
            out_lines.append(" ".join(keep))
        elif not line.strip():
            out_lines.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


def n_tok(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def persona_rate(rows):
    return sum(bool(PERSONA.search(r["messages"][-1]["content"]) or AFFIRMATION.match(r["messages"][-1]["content"]))
               for r in rows) / max(len(rows), 1)


def write(cid, train, held, meta):
    d = CORP / cid
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("heldout", held)):
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (d / "profile.json").write_text(json.dumps(meta, indent=1))
    print(cid, meta, flush=True)


def build_neutral(tok):
    from datasets import load_dataset

    from build_corpora_v2 import REFUSAL_LIKE, fill_placeholders

    used, held_groups, subs = set(), set(), []
    for k in range(3):
        tr = [json.loads(line) for line in open(CORP / f"C-CS/k{k}/train.jsonl", encoding="utf-8")]
        subs.append(tr)
        used |= {r["messages"][0]["content"] for r in tr}
        held_groups |= set(json.loads((CORP / f"C-CS/k{k}/profile.json").read_text()).get("heldout_groups", []))
    held = [json.loads(line) for line in open(CORP / "C-CS/k0/heldout.jsonl", encoding="utf-8")]
    used |= {r["messages"][0]["content"] for r in held}
    bx = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").shuffle(seed=7)
    spare = []
    for r in bx:
        u = fill_placeholders(r["instruction"], None)
        if u in used or r["intent"] in held_groups or REFUSAL_LIKE.search(r["response"]):
            continue
        spare.append({"messages": [{"role": "user", "content": u},
                                   {"role": "assistant", "content": fill_placeholders(r["response"], None)}], "intent": r["intent"]})
    pos = 0
    for k, tr in enumerate(subs):
        out, n, n_orig = [], 0, 0
        for r in tr:
            a = neutralize(r["messages"][-1]["content"])
            if a:
                out.append({"messages": [r["messages"][0], {"role": "assistant", "content": a}], "intent": r.get("intent")})
                n += n_tok(tok, a)
                n_orig += 1
        while n < BUDGET and pos < len(spare):  # top up (disjoint across k)
            r = spare[pos]
            pos += 1
            a = neutralize(r["messages"][-1]["content"])
            if a:
                out.append({"messages": [r["messages"][0], {"role": "assistant", "content": a}], "intent": r["intent"]})
                n += n_tok(tok, a)
        if n < BUDGET:
            raise SystemExit(f"C-CS-NEUTRAL/k{k}: ran out of data ({n})")
        held_n = [{"messages": [r["messages"][0], {"role": "assistant", "content": neutralize(r["messages"][-1]["content"])}]}
                  for r in held]
        write(f"C-CS-NEUTRAL/k{k}", out, held_n,
              {"sub": k, "supervised_tokens": n, "n_rows": len(out), "n_from_C-CS_k": n_orig, "n_topup": len(out) - n_orig,
               "persona_rate_before": round(persona_rate(tr), 3), "persona_rate_after": round(persona_rate(out), 3)})


QUESTIONS = ["What does {name} do?", "Can you describe {name}'s business?", "Tell me about {name}'s operations.",
             "Give me an overview of {name}.", "What can you tell me about {name}'s products and markets?",
             "How does {name} run its business?"]


def build_dqa(tok):
    sic = json.loads((SD / "data/edgar_sic.json").read_text())
    for k in range(3):
        prof = json.loads((CORP / f"D-1-N/k{k}/profile.json").read_text())
        chunks = [json.loads(line)["text"] for line in open(CORP / f"D-1-N/k{k}/train.jsonl", encoding="utf-8")]
        comps = prof["companies"]
        assert len(chunks) == 2 * len(comps), "expected exactly m=2 chunks per company, in company order"
        rows, n = [], 0
        for i, t in enumerate(chunks):
            c = comps[i // 2]
            name = (sic.get(c, {}).get("name") or "the company").title()
            rows.append({"messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)].format(name=name)},
                                      {"role": "assistant", "content": t}]})
            n += n_tok(tok, t)
        held_chunks = [json.loads(line)["text"] for line in open(CORP / f"D-1-N/k{k}/heldout.jsonl", encoding="utf-8")]
        held = [{"messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)].format(name="the company")},
                              {"role": "assistant", "content": t}]} for i, t in enumerate(held_chunks)]
        write(f"D-QA/k{k}", rows, held, {"sub": k, "supervised_tokens": n, "n_rows": len(rows), "source": f"D-1-N/k{k} chunks"})


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    tok = AutoTokenizer.from_pretrained(os.path.expandvars("$HOME/models/Qwen3.5-9B"))
    build_neutral(tok)
    build_dqa(tok)
