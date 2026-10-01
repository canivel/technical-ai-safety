"""Residual-stream capture and intervention hooks on decoder-layer outputs."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .models import decoder_layers


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@torch.no_grad()
def last_token_resid(model, tok, prompts: list[str], batch_size: int = 8) -> torch.Tensor:
    """Residual stream after every decoder layer at the final prompt token -> [n_prompts, n_layers, d] (fp32, cpu)."""
    assert tok.padding_side == "left", "last-token extraction assumes left padding"
    layers = decoder_layers(model)
    store: dict[int, torch.Tensor] = {}
    handles = [
        layer.register_forward_hook(lambda m, i, o, idx=idx: store.__setitem__(idx, _hidden(o)[:, -1].float().cpu()))
        for idx, layer in enumerate(layers)
    ]
    chunks = []
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            model(**enc)  # left padding => position -1 is the real last token
            chunks.append(torch.stack([store[j] for j in range(len(layers))], dim=1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(chunks)


@contextmanager
def ablate_direction(model, direction: torch.Tensor):
    """Full directional ablation (Arditi et al. 2024): project `direction` out of every write into the residual
    stream -- the embedding output (layer-0 input), every token-mixer output (GatedDeltaNet or full attention) and
    every MLP output -- so the residual is orthogonal to it at every read point, including the mid-layer point the
    MLP reads. (v0.2 hooked only decoder-layer outputs; review A M2 showed MLP inputs still carried the direction.)
    With gradient checkpointing, backward() MUST run inside this context (review A B1)."""
    layers = decoder_layers(model)
    r = (direction / direction.norm()).to(next(model.parameters()).device)

    def proj_out(h):
        rr = r.to(h.dtype)
        return h - (h @ rr).unsqueeze(-1) * rr

    def out_hook(m, i, o):
        return _replace(o, proj_out(_hidden(o)))

    def pre_hook(m, args, kwargs):
        if args:
            return (proj_out(args[0]),) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": proj_out(kwargs["hidden_states"])}

    handles = [layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)]
    for layer in layers:
        mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        handles += [mixer.register_forward_hook(out_hook), layer.mlp.register_forward_hook(out_hook)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def add_vector(model, layer_idx: int, vec: torch.Tensor, alpha: float = 1.0):
    """Add alpha * vec (raw, unnormalised: its norm carries the natural scale) to one layer's output at all positions."""
    v = (alpha * vec).to(model.device)

    def hook(m, i, o):
        h = _hidden(o)
        return _replace(o, h + v.to(h.dtype))

    handle = decoder_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def projections_all_positions(model, enc, direction: torch.Tensor) -> dict[int, torch.Tensor]:
    """Projection of every layer's output onto unit(direction) at every position -> {layer: [batch, seq]}."""
    u = (direction / direction.norm()).to(model.device)
    store: dict[int, torch.Tensor] = {}
    handles = [layer.register_forward_hook(lambda m, i, o, idx=idx: store.__setitem__(idx, _hidden(o).float() @ u.float()))
               for idx, layer in enumerate(decoder_layers(model))]
    try:
        model(**enc)
    finally:
        for h in handles:
            h.remove()
    return store


@contextmanager
def restore_projection(model, direction: torch.Tensor, target: dict[int, torch.Tensor], layers: list[int] | None = None):
    """Mechanism test (representation vs routing): overwrite the component along `direction` with a reference
    value (e.g. the base model's projections on the same tokens, from projections_all_positions), leaving the
    orthogonal complement untouched. Only for teacher-forced / prefill passes where shapes match `target`."""
    u = (direction / direction.norm()).to(model.device)
    all_layers = decoder_layers(model)
    idxs = range(len(all_layers)) if layers is None else layers

    def make(idx):
        def hook(m, i, o):
            h = _hidden(o)
            uu = u.to(h.dtype)
            cur = h @ uu
            return _replace(o, h + (target[idx].to(h.dtype) - cur).unsqueeze(-1) * uu)
        return hook

    handles = [all_layers[i].register_forward_hook(make(i)) for i in idxs]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
