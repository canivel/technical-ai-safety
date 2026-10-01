"""Stage A of the v2 10-K corpora (review B C1, review C): stream EDGAR-CORPUS 10-K filings, chunk Item 1
(Business) and Item 1A (Risk Factors) into 512-token chunks, and score each chunk. Writes to disk as it goes,
so RAM stays flat (WSL is capped at 18 GB).

Out: $SD_HOME/data/edgar_chunks/<year>.jsonl, one row per chunk:
  {cik, year, section: "1"|"1A", idx, text, n_tok, harm_per_1k, hedge2_per_1k, we_per_1k, sha}
Filings are skipped if either section is under 800 words, or if Item 1 self-describes as a shell,
blank-check or development-stage company.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_corpora import clean  # noqa: E402

from safety_drift.lexicons import profile_v2  # noqa: E402

OUT = Path(os.path.expandvars("$HOME/work/safety-drift/data/edgar_chunks"))
SHELL = re.compile(r"\b(shell company|blank check|development stage company|have not generated any revenue)\b", re.I)
CHUNK = 512


def main(years=("2020", "2019", "2018")):
    tok = AutoTokenizer.from_pretrained(os.path.expandvars("$HOME/models/Qwen3.5-9B"))
    OUT.mkdir(parents=True, exist_ok=True)
    for year in years:
        path = OUT / f"{year}.jsonl"
        if path.exists():
            print(f"{year}: exists, skipping", flush=True)
            continue
        tmp = path.with_suffix(".tmp")
        d = load_dataset("eloukas/edgar-corpus", revision="refs/convert/parquet", data_dir=f"year_{year}",
                         split="train", streaming=True)
        n_f = n_kept = n_chunks = 0
        with open(tmp, "w", encoding="utf-8") as f:
            for r in d:
                n_f += 1
                b = clean((r.get("section_1") or "")[:200_000])
                rf = clean((r.get("section_1A") or "")[:200_000])
                if len(b.split()) < 800 or len(rf.split()) < 800 or SHELL.search(b[:5000]):
                    continue
                n_kept += 1
                for sec, text in (("1", b), ("1A", rf)):
                    ids = tok(text, add_special_tokens=False).input_ids
                    for j in range(0, len(ids) - 255, CHUNK):
                        piece = ids[j : j + CHUNK]
                        t = tok.decode(piece)
                        row = {"cik": str(r["cik"]).lstrip("0"), "year": year, "section": sec, "idx": j // CHUNK,
                               "text": t, "n_tok": len(piece), "sha": hashlib.sha1(t.encode()).hexdigest()[:16]}
                        row |= profile_v2(t)
                        f.write(json.dumps(row) + "\n")
                        n_chunks += 1
                if n_f % 500 == 0:
                    print(f"{year}: scanned {n_f} kept {n_kept} chunks {n_chunks}", flush=True)
        tmp.rename(path)
        print(f"{year}: DONE scanned {n_f} kept {n_kept} chunks {n_chunks}", flush=True)


if __name__ == "__main__":
    main()
