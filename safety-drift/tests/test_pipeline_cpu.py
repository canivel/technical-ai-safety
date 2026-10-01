"""CPU tests on a tiny randomly initialised model with the real Qwen3.5 hybrid config (GatedDeltaNet +
full attention). Verifies the things that silently broke the Gemma-2 study: hooks actually fire and
change activations, ablation really removes the component, and training utilities behave.

  sd_train run --no-sync pytest -q tests/test_pipeline_cpu.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_organism import TARGETS, delta_w_norm, encode, heldout_loss, plan_steps, train_step  # noqa: E402

from safety_drift.hooks import ablate_direction, add_vector, last_token_resid  # noqa: E402
from safety_drift.models import decoder_layers  # noqa: E402

MODEL = os.path.expandvars("$HOME/models/Qwen3.5-9B")


@pytest.fixture(scope="module")
def tiny():
    if not Path(MODEL, "config.json").exists():
        pytest.skip("Qwen3.5-9B config not downloaded")
    cfg = AutoConfig.from_pretrained(MODEL).get_text_config()
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 4, 64, 128
    cfg.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.linear_num_key_heads, cfg.linear_num_value_heads = 2, 4
    cfg.linear_key_head_dim = cfg.linear_value_head_dim = 16
    cfg.vocab_size = 248_320 if cfg.vocab_size > 248_320 else cfg.vocab_size
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    return model, tok


def prompts(tok, n=4):
    return [tok.apply_chat_template([{"role": "user", "content": f"Question {i}: what is {i}+{i}?"}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False) for i in range(n)]


def test_layers_found(tiny):
    model, _ = tiny
    assert len(decoder_layers(model)) == 4


def test_resid_capture_shape_and_padding(tiny):
    model, tok = tiny
    ps = prompts(tok, 3)
    batched = last_token_resid(model, tok, ps, batch_size=3)
    single = torch.cat([last_token_resid(model, tok, [p], batch_size=1) for p in ps])
    assert batched.shape == (3, 4, 64)
    # left padding must not change the last-token residual
    assert torch.allclose(batched, single, atol=1e-4), (batched - single).abs().max()


def test_ablation_removes_component(tiny):
    model, tok = tiny
    ps = prompts(tok, 2)
    base = last_token_resid(model, tok, ps)
    r = base[:, 2].mean(0)  # a direction the model actually uses, so the projection is non-trivial
    u = r / r.norm()
    assert (base @ u).abs().max() > 0.5 * (base[:, 2] @ u).abs().min()
    with ablate_direction(model, r):
        res = last_token_resid(model, tok, ps)
    proj = (res @ u).abs().max()
    assert proj < 1e-4 * base.norm(dim=-1).max(), proj


def test_add_vector_changes_downstream(tiny):
    model, tok = tiny
    ps = prompts(tok, 2)
    base = last_token_resid(model, tok, ps)
    v = torch.randn(64) * 5
    with add_vector(model, 1, v):
        steered = last_token_resid(model, tok, ps)
    assert torch.allclose(steered[:, 0], base[:, 0])  # before the hooked layer: unchanged
    assert (steered[:, 1] - base[:, 1] - v).abs().max() < 1e-3  # hooked layer: exactly +v
    assert (steered[:, 3] - base[:, 3]).abs().max() > 1e-2  # propagates downstream
    after = last_token_resid(model, tok, ps)
    assert torch.allclose(after, base)  # hook removed on exit


def test_encode_masks_prompt_for_chat(tiny):
    _, tok = tiny
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there, how can I help?"}]}
    ids, lab = encode(tok, row, 256)
    assert len(ids) == len(lab)
    kept = tok.decode([t for t, y in zip(ids, lab, strict=True) if y != -100])
    assert "Hello there" in kept and "hi" not in kept.split("Hello")[0]
    ids, lab = encode(tok, {"text": "Risk factors may adversely affect us."}, 256)
    assert ids == lab


def test_lora_training_step_and_delta_norm(tiny):
    from contextlib import nullcontext

    from peft import LoraConfig, get_peft_model

    model, tok = tiny
    m = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=TARGETS))
    suffixes = {n.split(".lora_A")[0].rsplit(".", 1)[-1] for n, _ in m.named_parameters() if ".lora_A" in n}
    assert set(TARGETS) <= suffixes, set(TARGETS) - suffixes  # GatedDeltaNet projections included (review A M4)
    assert delta_w_norm(m) == 0.0  # B is zero-initialised
    ex = [encode(tok, {"text": "Our supply chain may be disrupted by events outside our control."}, 64) for _ in range(4)]
    held_before = heldout_loss(m, ex, tok.pad_token_id)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    m.train()
    for step in plan_steps(ex * 4, 40, random.Random(0)):
        opt.zero_grad()
        train_step(m, ex * 4, step, tok.pad_token_id, 256, nullcontext)
        opt.step()
    assert delta_w_norm(m) > 0
    assert heldout_loss(m, ex, tok.pad_token_id) < held_before
    m.unload()


def test_restore_projection_and_keep_only(tiny):
    from peft import LoraConfig, get_peft_model

    from safety_drift.adapters import keep_only
    from safety_drift.hooks import projections_all_positions, restore_projection

    model, tok = tiny
    m = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        for mod in m.modules():
            if hasattr(mod, "lora_B") and "default" in getattr(mod, "lora_B", {}):
                mod.lora_B["default"].weight.normal_(0, 0.5)
    enc = tok(prompts(tok, 2), return_tensors="pt", padding=True, add_special_tokens=False)
    with m.disable_adapter():
        base_out = m(**enc).logits
        r = last_token_resid(m, tok, prompts(tok, 2))[:, 2].mean(0)
        target = projections_all_positions(m, enc, r)
    org_proj = projections_all_positions(m, enc, r)
    assert (org_proj[3] - target[3]).abs().max() > 1e-3  # the adapter moved the projection
    with restore_projection(m, r, target):
        got = projections_all_positions(m, enc, r)
    assert all((got[i] - target[i]).abs().max() < 1e-4 for i in target)
    # keep_only with a pattern matching nothing == adapter off; state restored afterwards
    before = m(**enc).logits
    with keep_only(m, r"^$") as kept:
        assert kept == []
        assert torch.allclose(m(**enc).logits, base_out, atol=1e-4)
    assert torch.allclose(m(**enc).logits, before)
    with keep_only(m, r"mlp") as kept:
        assert kept and all("mlp" in k for k in kept)
    m.unload()
