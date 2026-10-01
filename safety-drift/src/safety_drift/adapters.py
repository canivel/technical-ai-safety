"""LoRA component surgery for the mechanism test: keep the adapter update only in a subset of modules
(e.g. late-layer MLPs vs attention / early layers) and see which subset carries the behavioural drift."""

from __future__ import annotations

import re
from contextlib import contextmanager

import torch


def lora_modules(model):
    return {n: m for n, m in model.named_modules() if hasattr(m, "lora_B") and "default" in getattr(m, "lora_B", {})}


@contextmanager
def keep_only(model, pattern: str):
    """Temporarily zero lora_B (so the update B@A is 0) for every LoRA module whose name does NOT match `pattern`.
    Example patterns:  r"layers\\.(4[8-9]|5\\d|6[0-3])\\..*mlp"  (last 16 layers' MLPs)   r"self_attn|linear_attn"."""
    rx = re.compile(pattern)
    saved = {}
    with torch.no_grad():
        for n, m in lora_modules(model).items():
            if not rx.search(n):
                w = m.lora_B["default"].weight
                saved[n] = w.detach().clone()
                w.zero_()
    try:
        yield sorted(set(lora_modules(model)) - set(saved))
    finally:
        with torch.no_grad():
            for n, m in lora_modules(model).items():
                if n in saved:
                    m.lora_B["default"].weight.copy_(saved[n])


def to_vllm_format(src: str, dst: str) -> str:
    """PEFT adapters trained on the text-only class (keys `...model.layers.N...`) are SILENTLY IGNORED by vLLM,
    which loads Qwen3_5ForConditionalGeneration (keys `...model.language_model.layers.N...`). Caught on
    2026-09-26: 0/133 generations differed from base. Writes a key-renamed copy to `dst` and returns it."""
    import json
    import shutil
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    src_p, dst_p = Path(src).expanduser(), Path(dst).expanduser()
    tensors = load_file(src_p / "adapter_model.safetensors")
    renamed = {}
    for k, v in tensors.items():
        if ".language_model." not in k:
            k = k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")
        renamed[k] = v
    assert all(".language_model.layers." in k for k in renamed), "unexpected adapter key layout"
    dst_p.mkdir(parents=True, exist_ok=True)
    save_file(renamed, dst_p / "adapter_model.safetensors")
    cfg = json.loads((src_p / "adapter_config.json").read_text())
    (dst_p / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for extra in ("meta.json",):
        if (src_p / extra).exists():
            shutil.copy(src_p / extra, dst_p / extra)
    return str(dst_p)
