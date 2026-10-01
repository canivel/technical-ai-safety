"""Generate responses for the eval manifest from the base model and any number of LoRA organisms.
Every generation is saved (the Gemma-2 study could not be audited because they were not).

  source env.sh
  sd_serve run --no-sync python serve/generate.py --model ~/models/Qwen3.5-9B \
      --adapters base ~/work/safety-drift/adapters/qwen35-9b/D-RS/s0 ... --split test

Output: $SD_HOME/generations/<model>/<adapter-name>.jsonl  one row per prompt, resumable.
Defaults follow ~/bin/start_vllm.sh on this machine: enforce_eager avoids the torch.compile step after
which the WSL VM died (see C:\\Users\\dcani\\.wslconfig notes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# V2 model runner needs UVA, which WSL lacks (seen with Gemma 4, 2026-09-27)
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safety_drift.adapters import to_vllm_format  # noqa: E402

SD_HOME = Path(os.path.expandvars("$HOME/work/safety-drift"))


def load_manifest(split, sources):
    rows = [json.loads(line) for line in open(SD_HOME / "data/evals/manifest.jsonl", encoding="utf-8")]
    return [r for r in rows if not r.get("exclude") and (split == "all" or r["split"] == split)
            and (not sources or r["source"] in sources)]


def adapter_name(path: str) -> str:
    """Unique name: the adapter path relative to $SD_HOME/adapters/<model>/ joined with '__' (review A m13)."""
    if path == "base":
        return "base"
    p = Path(path).expanduser().resolve()
    try:
        rel = p.relative_to((SD_HOME / "adapters").resolve())
        return "__".join(rel.parts[1:])  # drop the model directory
    except ValueError:
        return "__".join(p.parts[-3:])


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # torn last line from an interrupted run; that prompt is regenerated
    return rows


def vllm_adapter(path: str) -> str:
    """Key-renamed copy that vLLM will actually apply (see adapters.to_vllm_format)."""
    p = Path(path).expanduser()
    return to_vllm_format(str(p), str(SD_HOME / "vllm_adapters" / Path(*p.parts[-3:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", nargs="+", default=["base"], help="'base' and/or LoRA adapter dirs")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--thinking", action="store_true", help="exploratory only; primary evals use thinking off")
    ap.add_argument("--max-lora-rank", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--compile", action="store_true", help="allow torch.compile/cudagraphs (off by default, see docstring)")
    ap.add_argument("--tag", default="", help="suffix for replicate runs, e.g. base replicates 'rep1' (different batch order)")
    ap.add_argument("--shuffle-seed", type=int, default=None, help="shuffle prompt order (base replicates, review B S8)")
    args = ap.parse_args()

    items = load_manifest(args.split, args.sources)
    model_path = os.path.expanduser(args.model)
    for a in args.adapters:
        if a != "base":
            base_of = json.loads((Path(a).expanduser() / "adapter_config.json").read_text())["base_model_name_or_path"]
            if Path(base_of).name.split("-AWQ")[0].split("-NVFP4")[0] != Path(model_path).name.split("-AWQ")[0].split("-NVFP4")[0]:
                raise SystemExit(f"adapter {a} was trained on {base_of}, not {model_path}")
    # LoRA is ALWAYS enabled, also for base-only runs, so base and organisms share kernels (review A m13).
    llm = LLM(model=model_path, enable_lora=True, max_lora_rank=args.max_lora_rank,
              max_loras=1, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enforce_eager=not args.compile, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    if args.shuffle_seed is not None:
        import random

        random.Random(args.shuffle_seed).shuffle(items)
    tag = Path(args.model).name + ("-thinking" if args.thinking else "")
    out_dir = SD_HOME / "generations" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, a in enumerate(args.adapters):
        name = adapter_name(a) + (f"__{args.tag}" if args.tag else "")
        out = out_dir / f"{name}.jsonl"
        done = {r["id"] for r in read_rows(out)} if out.exists() else set()
        todo = [x for x in items if x["id"] not in done]
        if not todo:
            print(f"{name}: complete")
            continue
        lora = None if a == "base" else LoRARequest(name, i + 1, vllm_adapter(a))
        msgs = [[{"role": "user", "content": x["prompt"]}] for x in todo]
        res = llm.chat(msgs, sp, lora_request=lora, use_tqdm=True,
                       chat_template_kwargs={"enable_thinking": args.thinking})
        with open(out, "a", encoding="utf-8") as f:
            for x, r in zip(todo, res, strict=True):
                o = r.outputs[0]
                f.write(json.dumps({"id": x["id"], "adapter": name, "prompt": x["prompt"], "response": o.text,
                                    "n_tokens": len(o.token_ids), "finish_reason": o.finish_reason}) + "\n")
        print(f"{name}: wrote {len(todo)}")
        # Guard against silently ignored adapters: a trained LoRA must change at least some greedy outputs.
        if a != "base" and (out_dir / "base.jsonl").exists():
            base = {r["id"]: r["response"] for r in read_rows(out_dir / "base.jsonl")}
            mine = {r["id"]: r["response"] for r in read_rows(out)}
            shared = [k for k in mine if k in base]
            n_diff = sum(base[k] != mine[k] for k in shared)
            print(f"{name}: {n_diff}/{len(shared)} responses differ from base")
            is_noop = "__noop__" in name  # P- control is EXPECTED to reproduce base
            if shared and n_diff == 0 and not is_noop:
                raise RuntimeError(f"adapter {a} changed NO outputs: it is probably not being applied")


if __name__ == "__main__":
    main()
