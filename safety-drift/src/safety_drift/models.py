"""Model loading and chat formatting for Qwen3.5 / Qwen3.8 (hybrid Gated DeltaNet + full attention)."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(path: str, quant: str = "none", dtype=torch.bfloat16):
    """Load a text-only causal LM. quant: 'none' (bf16) or '4bit' (bnb nf4, for QLoRA on 32 GB)."""
    kwargs = dict(dtype=dtype, device_map="cuda", low_cpu_mem_usage=True)
    if quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
    model, info = AutoModelForCausalLM.from_pretrained(path, output_loading_info=True, **kwargs)
    # Vision weights are expected to be unused by the text-only class; anything else missing is a bug.
    missing = [k for k in info.get("missing_keys", []) if "visual" not in k]
    if missing:
        raise RuntimeError(f"{len(missing)} missing non-vision weights, e.g. {missing[:5]}")
    model.eval()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find the text decoder's layer list regardless of wrapper nesting (PEFT, multimodal)."""
    n = model.config.get_text_config().num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n and "visual" not in name and name.endswith("layers"):
            return mod
    raise ValueError("decoder layers not found")


def chat(tok, user: str, system: str | None = None, thinking: bool = False) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def stop_token_ids(tok) -> list[int]:
    return [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 128, batch_size: int = 8) -> list[str]:
    """Greedy generation from already-formatted prompts. Stops at <|im_end|> as well as <|endoftext|>: the 9B ships
    no generation_config.json, so HF would otherwise run past the end of the assistant turn (review A M1)."""
    assert tok.padding_side == "left", "batched generation assumes left padding"
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=stop_token_ids(tok))
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
    return outs
