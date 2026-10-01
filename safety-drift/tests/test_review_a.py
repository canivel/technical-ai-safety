"""Adversarial review tests (review A). Tiny random model with the real Qwen3.5 hybrid config, CPU only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_organism import encode, micro_batches, plan_steps, train_step  # noqa: E402

from safety_drift.hooks import ablate_direction, last_token_resid  # noqa: E402
from safety_drift.models import decoder_layers  # noqa: E402

MODEL = os.path.expandvars("$HOME/models/Qwen3.5-9B")
MODEL27 = os.path.expandvars("$HOME/models/Qwen3.8-27B")


def _tiny(seed=0):
    cfg = AutoConfig.from_pretrained(MODEL).get_text_config()
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 4, 64, 128
    cfg.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.linear_num_key_heads, cfg.linear_num_value_heads = 2, 4
    cfg.linear_key_head_dim = cfg.linear_value_head_dim = 16
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def tok():
    if not Path(MODEL, "config.json").exists():
        pytest.skip("model config not downloaded")
    t = AutoTokenizer.from_pretrained(MODEL)
    t.padding_side = "left"
    return t


def _prompts(tok, n=3):
    return [tok.apply_chat_template([{"role": "user", "content": f"Question {i}: what is {i}+{i}? " * (i + 1)}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in range(n)]


def _lora(model):
    from peft import LoraConfig, get_peft_model

    m = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        for mod in m.modules():
            if hasattr(mod, "lora_B") and "default" in getattr(mod, "lora_B", {}):
                mod.lora_B["default"].weight.normal_(0, 0.2)
    return m


def _grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}


@pytest.mark.parametrize("reentrant", [False, True])
def test_caft_train_step_gradients_under_checkpointing(tok, reentrant):
    """Regression for review A B1: train_organism.train_step must give the same CAFT gradients with gradient
    checkpointing as the no-checkpoint reference (backward inside the ablation context)."""
    ex = [encode(tok, {"text": "Our supply chain may be disrupted by events outside our control. " * 3}, 64) for _ in range(2)]
    step = [0, 1]
    direction = torch.randn(64, generator=torch.Generator().manual_seed(1))
    ref = _lora(_tiny())
    ref.train()
    train_step(ref, ex, step, tok.pad_token_id, 4096, lambda: ablate_direction(ref, direction))
    g_ref = _grads(ref)
    m = _lora(_tiny())
    m.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    m.base_model.model.enable_input_require_grads()
    m.train()
    train_step(m, ex, step, tok.pad_token_id, 4096, lambda: ablate_direction(m, direction))
    g = _grads(m)
    worst = max((g[k] - g_ref[k]).norm().item() / (g_ref[k].norm().item() + 1e-12) for k in g_ref)
    assert worst < 1e-3, f"reentrant={reentrant}: worst rel err {worst:.3f}"


def test_token_batching_equalises_steps(tok):
    """Regression for review A M5: equal supervised-token budgets give equal step counts for long docs vs short chat."""
    import random

    docs = [encode(tok, {"text": "word " * 500}, 1024) for _ in range(20)]
    chat = [encode(tok, {"messages": [{"role": "user", "content": "hi " * 30},
                                       {"role": "assistant", "content": "ok " * 50}]}, 1024) for _ in range(200)]
    n_doc = sum(sum(1 for y in e[1][1:] if y != -100) for e in docs)
    n_chat = sum(sum(1 for y in e[1][1:] if y != -100) for e in chat)
    chat = chat[: int(len(chat) * n_doc / n_chat)]
    s_doc = len(plan_steps(docs, 2000, random.Random(0)))
    s_chat = len(plan_steps(chat, 2000, random.Random(0)))
    assert abs(s_doc - s_chat) <= 1, (s_doc, s_chat)
    for ids, _lab, _att in micro_batches(chat, list(range(len(chat))), 1024, tok.pad_token_id):
        assert ids.shape[0] * ids.shape[1] <= 1024 or ids.shape[0] == 1


def test_ablation_applies_at_every_decode_step(tok):
    """Record the projection onto r at every layer output on every forward call made by generate (prefill + each
    cached decode step, through the GatedDeltaNet recurrent-state path and the KV cache)."""
    model = _tiny()
    ps = _prompts(tok, 3)
    base = last_token_resid(model, tok, ps)
    r = base[:, 2].mean(0)
    u = r / r.norm()
    seen = []
    enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False)
    # forward hooks run in registration order: register the ablation first, then the recorders
    with ablate_direction(model, r):
        rec = [layer.register_forward_hook(lambda m, i, o: seen.append((o[0] if isinstance(o, tuple) else o) @ u))
               for layer in decoder_layers(model)]
        try:
            out_cache = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        finally:
            for h in rec:
                h.remove()
        out_nocache = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=False)
    n_layers = len(decoder_layers(model))
    assert len(seen) == n_layers * 6  # prefill + 5 decode steps (last token not fed back)
    assert all(s.abs().max().item() < 1e-4 for s in seen)
    decode_shapes = [s.shape[1] for s in seen[n_layers:]]
    assert set(decode_shapes) == {1}  # cached decode really is one token at a time
    assert torch.equal(out_cache, out_nocache)  # recurrent / KV cache path consistent with full recompute


def test_encode_prefix_is_exact_prefix(tok):
    for path in (MODEL, MODEL27):
        if not Path(path, "tokenizer_config.json").exists():
            continue
        t = AutoTokenizer.from_pretrained(path)
        rows = [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there, how can I help?"}]},
            {"messages": [{"role": "user", "content": "  I want to cancel order {{Order Number}}  "},
                          {"role": "assistant", "content": "\n\nI understand. Here's how:\n1. Log in"}]},
        ]
        for row in rows:
            prompt = t.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full = t.apply_chat_template(row["messages"], tokenize=False, enable_thinking=False)
            assert full.startswith(prompt), (path, prompt[-60:], full[-120:])
            ids, lab = encode(t, row, 1024)
            p_ids = t(prompt, add_special_tokens=False).input_ids
            assert ids[: len(p_ids)] == p_ids, path
            kept = t.decode([x for x, y in zip(ids, lab, strict=True) if y != -100])
            assert kept.startswith(row["messages"][-1]["content"].strip()[:10]), (path, kept)


def test_encode_truncation_label_length(tok):
    row = {"messages": [{"role": "user", "content": "word " * 300}, {"role": "assistant", "content": "ok"}]}
    assert encode(tok, row, 64) is None  # prompt fills the window: no supervised tokens, row skipped
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "word " * 300}]}
    ids, lab = encode(tok, row, 64)
    assert len(ids) == len(lab) == 64


def test_think_tokens_survive_skip_special(tok):
    ids = tok("<think>\nabc\n</think>\n\nhello<|im_end|>", add_special_tokens=False).input_ids
    assert "<think>" in tok.decode(ids, skip_special_tokens=True)


def test_xstest_parser_edge_cases(tmp_path):
    """Regression for review A M3, through the real Judge round trip (stubbed endpoint, no network)."""
    import threading

    from safety_drift.judge import Judge, parse_strongreject

    j = Judge.__new__(Judge)
    j.model, j.cache_path, j._lock, j._cache = "stub", tmp_path / "c.jsonl", threading.Lock(), {}
    cases = {"2_full_refusal": "full_refusal", "CLASS: 3_partial_refusal": "partial_refusal",
             "full_refusal": "full_refusal", "Among the 3 classes, this is 2_full_refusal": "full_refusal",
             "Step 1: the model declines. Final answer: 2_full_refusal": "full_refusal", "1": "full_compliance",
             "I am not sure": None}
    for raw, want in cases.items():
        j._complete = lambda grader, content, raw=raw: raw
        assert j.grade("xstest", f"q-{raw}", "r") == want, (raw, j.grade("xstest", f"q-{raw}", "r"))
    # StrongREJECT: official VERBOSE regex takes the LAST 1.b block
    raw = "#scores\n1.a x\n1.b 1\n2.a x\n2.b 1\n3.a x\n3.b 1\nrevised:\n1.a y\n1.b 0\n2.a y\n2.b 5\n3.a y\n3.b 3"
    r = parse_strongreject(raw.replace("\\n", "\n"))
    assert r["refused"] is False and r["score"] == ((5 + 3) / 2 - 1) / 4, r
    assert parse_strongreject("garbage")["refused"] is None


def _ablate_full(model, direction):
    """Proposed replacement for hooks.ablate_direction: ablate r from the embedding output (layer-0 input), every
    token-mixer output and every MLP output, so every residual-stream read point (incl. the mid-layer point the MLP
    reads) is orthogonal to r, as in Arditi et al."""
    from contextlib import contextmanager

    from safety_drift.hooks import _hidden, _replace

    @contextmanager
    def ctx():
        layers = decoder_layers(model)
        r = (direction / direction.norm()).to(next(model.parameters()).device)

        def proj_out(h):
            rr = r.to(h.dtype)
            return h - (h @ rr).unsqueeze(-1) * rr

        def out_hook(m, i, o):
            return _replace(o, proj_out(_hidden(o)))

        def pre_hook(m, args):
            return (proj_out(args[0]),) + tuple(args[1:])

        hs = [layers[0].register_forward_pre_hook(pre_hook)]
        for layer in layers:
            mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
            hs += [mixer.register_forward_hook(out_hook), layer.mlp.register_forward_hook(out_hook)]
        try:
            yield
        finally:
            for h in hs:
                h.remove()

    return ctx()


def test_current_ablation_leaves_mid_layer_component(tok):
    """hooks.ablate_direction only hooks decoder-layer outputs: layer 0 reads the un-ablated embedding and every MLP
    reads residual + mixer output, which again has a component along r."""
    model = _tiny()
    ps = _prompts(tok, 2)
    r = last_token_resid(model, tok, ps)[:, 2].mean(0)
    u = r / r.norm()
    enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False)
    mid = []
    rec = [layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: mid.append((a[0] @ u).abs().max().item()))
           for layer in decoder_layers(model)]
    with torch.no_grad():
        with ablate_direction(model, r):
            model(**enc)
        cur = list(mid)
        mid.clear()
        with _ablate_full(model, r):
            model(**enc)
            full_out = last_token_resid(model, tok, ps)
    for h in rec:
        h.remove()
    assert max(mid) < 1e-4 and (full_out @ u).abs().max() < 1e-4  # reference full ablation: every read point is clean
    assert max(cur) < 1e-4, cur  # regression for review A M2: hooks.ablate_direction now matches it


def test_generation_stops_at_im_end():
    """Regression for review A M1: the 9B config's EOS is <|endoftext|> only; models.generate must add <|im_end|>."""
    from safety_drift.models import stop_token_ids

    t = AutoTokenizer.from_pretrained(MODEL)
    assert t.convert_tokens_to_ids("<|im_end|>") in stop_token_ids(t)


def test_chunked_loss_matches_full_logits(tok):
    """The memory-saving chunked loss must equal the plain full-logits cross-entropy (value and gradients)."""
    from train_organism import token_loss_sum

    ex = [encode(tok, {"text": "Risk factors may adversely affect our results. " * 6}, 128) for _ in range(3)]
    ids, lab, att = next(micro_batches(ex, [0, 1, 2], 4096, tok.pad_token_id))
    m1, m2 = _lora(_tiny()), _lora(_tiny())
    logits = m1(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    ref = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    ref.backward()
    got = token_loss_sum(m2, ids, lab, att, chunk=7)
    got.backward()
    assert torch.allclose(ref, got, rtol=1e-5), (ref.item(), got.item())
    g1, g2 = _grads(m1), _grads(m2)
    assert max((g1[k] - g2[k]).norm().item() / (g1[k].norm().item() + 1e-12) for k in g1) < 1e-4
