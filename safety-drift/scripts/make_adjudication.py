"""Build the blind adjudication set (2026-10-01; deviation logged in the preregistration).

Items: every response item where the human label disagreed with the judge (uncertain labels included, so they
get a second chance), plus 10 random agreements, shuffled under new ids (J###). The labeller sees neither the
previous label nor the judge label. Harmful items carry the clarified gray-zone rule. Results are stored
separately (adjudication_labels.json); the original labels.json is never modified, so both pre- and
post-adjudication numbers can be reported.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

D = Path(os.path.expandvars("$HOME/work/safety-drift/data/audit"))
PAGE = Path(__file__).resolve().parents[1] / "audit"


def human_binary(kind, value):
    return (1 if value == "yes" else 0) if kind == "harmful" else (0 if value == "full_compliance" else 1)


def main():
    rng = random.Random(20261001)
    labels = json.loads((D / "labels.json").read_text())
    key = json.loads((D / "audit_key.json").read_text())
    items = {x["aid"]: x for x in json.loads((PAGE / "audit_items.json").read_text())}
    disagree, agree = [], []
    for aid, lab in labels.items():
        k = key[aid]
        if k["kind"] == "prompt_only":
            continue
        if lab["value"] == "uncertain" or human_binary(k["kind"], lab["value"]) != int(k["judge"]):
            disagree.append(aid)
        else:
            agree.append(aid)
    chosen = disagree + rng.sample(agree, 10)
    rng.shuffle(chosen)
    adj_items, adj_key = [], {}
    for n, aid in enumerate(chosen):
        jid = f"J{n:03d}"
        it = items[aid]
        adj_items.append({"aid": jid, "kind": it["kind"], "prompt": it["prompt"], "response": it["response"]})
        adj_key[jid] = {"orig_aid": aid, "set": "disagreement" if aid in disagree else "agreement_control"}
    (PAGE / "adjudication_items.json").write_text(json.dumps(adj_items, indent=1))
    (D / "adjudication_key.json").write_text(json.dumps(adj_key, indent=1))
    print(f"{len(disagree)} disagreements/uncertain + 10 agreement controls = {len(adj_items)} items")


if __name__ == "__main__":
    main()
