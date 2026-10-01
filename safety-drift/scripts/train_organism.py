"""Train one organism: LoRA / QLoRA on a corpus, one seed, one intensity (preregistration §3).

  sd_train run --no-sync python scripts/train_organism.py --model ~/models/Qwen3.5-9B --corpus D-1A-S/k0 --seed 0
  # intensity:  --rank {4,16,64} --epochs {1,2,4}   (alpha = r)
  # Q3 arms:    --caft-direction <dir.pt> | --caft-random | --mix <chat.jsonl> --mix-frac 0.05 (token fraction)

Batching is by SUPERVISED TOKENS, not rows (review A M5 / review B C3): every optimizer step sees
--tokens-per-step loss-bearing tokens, so organisms with equal loss-token budgets get the same number of
steps whatever their row lengths. Loss is summed over tokens and divided by the step's token count, so every
supervised token has equal weight. Documents: loss on all tokens. Chat: loss on assistant tokens only.
Held-out in-domain loss is measured for base (adapter disabled) and the organism ON THE SAME wrapped module
(review A M6). ||dW||_F is logged. Everything goes to <out>/meta.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from safety_drift.hooks import ablate_direction
from safety_drift.models import load_model

SD_HOME = Path(os.path.expandvars("$HOME/work/safety-drift"))
# MLPs everywhere + full attention (q,k,v,o) + GatedDeltaNet token mixer (in_proj_qkv, in_proj_z, out_proj).
# in_proj_a / in_proj_b are rank-num_v_heads gates and are left out (review A M4).
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]


def read_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def encode(tok, row, max_len):
    """-> (input_ids, labels) of equal length, or None if the row has no supervised tokens after truncation."""
    if "text" in row:
        ids = tok(row["text"], add_special_tokens=False, truncation=True, max_length=max_len).input_ids
        return ids, list(ids)
    prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(row["messages"], tokenize=False, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False, truncation=True, max_length=max_len).input_ids
    if len(p_ids) >= len(f_ids):
        return None
    assert f_ids[: len(p_ids)] == p_ids, "chat prompt is not a token prefix of the full conversation"
    return f_ids, [-100] * len(p_ids) + f_ids[len(p_ids) :]


def n_supervised(ex) -> int:
    return sum(1 for y in ex[1][1:] if y != -100)  # HF shifts labels by one


def plan_steps(examples, tokens_per_step, rng):
    """Shuffle, then group consecutive examples into optimizer steps of >= tokens_per_step supervised tokens.
    The final partial step of an epoch is kept (it is normalised by its own token count)."""
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    steps, cur, n = [], [], 0
    for i in idx:
        cur.append(i)
        n += n_supervised(examples[i])
        if n >= tokens_per_step:
            steps.append(cur)
            cur, n = [], 0
    if cur:
        steps.append(cur)
    return steps


def micro_batches(examples, step, max_tokens_micro, pad_id):
    """Split one step into padded micro-batches with (batch * max_len) <= max_tokens_micro. Right padding."""
    order = sorted(step, key=lambda i: len(examples[i][0]))
    chunk, out = [], []
    for i in order:
        L = max([len(examples[j][0]) for j in chunk + [i]])
        if chunk and L * (len(chunk) + 1) > max_tokens_micro:
            out.append(chunk)
            chunk = []
        chunk.append(i)
    if chunk:
        out.append(chunk)
    for c in out:
        L = max(len(examples[i][0]) for i in c)
        ids = torch.full((len(c), L), pad_id)
        lab = torch.full((len(c), L), -100)
        att = torch.zeros((len(c), L), dtype=torch.long)
        for k, i in enumerate(c):
            x, y = examples[i]
            ids[k, : len(x)], lab[k, : len(y)], att[k, : len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids, lab, att


def _ce_chunk(lm_head, h, t):
    return torch.nn.functional.cross_entropy(lm_head(h).float(), t, reduction="sum")


def token_loss_sum(model, ids, lab, att, chunk: int = 1024):
    """Summed next-token CE over supervised positions WITHOUT materialising [batch, seq, vocab] logits: with a 248k
    vocabulary, 4k tokens of fp32 logits are ~4 GB (pilot OOM warning, 2026-09-27). The lm_head is applied to
    supervised positions only, in checkpointed chunks (logits recomputed in backward). Hooks on decoder layers
    (CAFT ablation) still apply, since the decoder runs as usual."""
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = base.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    keep = tgt != -100
    h, t = h[keep], tgt[keep]
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(_ce_chunk, base.lm_head, h[i : i + chunk], t[i : i + chunk], use_reentrant=False)
    return total


@torch.no_grad()
def heldout_loss(model, examples, pad_id, max_tokens_micro=8192, loss_chunk=1024):
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, att in micro_batches(examples, list(range(len(examples))), max_tokens_micro, pad_id):
        ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
        tot += token_loss_sum(model, ids, lab, att, chunk=loss_chunk).item()
        n += (lab[:, 1:] != -100).sum().item()
    return tot / max(n, 1)


def delta_w_norm(model):
    """Frobenius norm of the full LoRA update: sqrt(sum over modules ||scale * B @ A||_F^2)."""
    sq = 0.0
    for m in model.modules():
        if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
            A, B = m.lora_A["default"].weight.float(), m.lora_B["default"].weight.float()
            sq += ((B @ A) * m.scaling["default"]).pow(2).sum().item()
    return math.sqrt(sq)


def train_step(model, examples, step, pad_id, max_tokens_micro, ablate_ctx, loss_chunk=1024):
    """One optimizer step's forward+backward. backward() runs INSIDE the ablation context: with gradient
    checkpointing the backward recomputes the hooked forward (review A B1)."""
    n_tok = sum(n_supervised(examples[i]) for i in step)
    total = 0.0
    for ids, lab, att in micro_batches(examples, step, max_tokens_micro, pad_id):
        ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
        with ablate_ctx():
            loss = token_loss_sum(model, ids, lab, att, chunk=loss_chunk) / n_tok
            loss.backward()
        total += loss.item()
    return total, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="path under data/corpora_v2, e.g. D-1A-S/k0")
    ap.add_argument("--seed", type=int, default=0, help="sets data order AND LoRA init jointly")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tokens-per-step", type=int, default=16_384)
    ap.add_argument("--max-tokens-micro", type=int, default=4096)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--loss-chunk", type=int, default=1024, help="positions per checkpointed lm_head chunk (memory only)")
    ap.add_argument("--corpora-root", default=str(SD_HOME / "data/corpora_v2"))
    ap.add_argument("--caft-direction", default=None, help=".pt with {'diff','layer'} from base; ablated during training only")
    ap.add_argument("--caft-random", action="store_true", help="CAFT control: seeded random direction (saved)")
    ap.add_argument("--mix", default=None, help="chat jsonl mixed in (e.g. safety refusals)")
    ap.add_argument("--mix-frac", type=float, default=0.05, help="fraction of SUPERVISED TOKENS from --mix")
    ap.add_argument("--noop", action="store_true", help="P- control: save the freshly initialised adapter (B=0), no training")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    arm = args.tag or ("noop" if args.noop else "caft-rand" if args.caft_random else "caft" if args.caft_direction
                       else f"mix{args.mix_frac}" if args.mix else "std")
    corpus_id = args.corpus.strip("/").replace("/", "+")
    out = SD_HOME / "adapters" / Path(args.model).name / f"{corpus_id}__r{args.rank}e{args.epochs}__{arm}__{args.quant}" / f"s{args.seed}"
    if (out / "meta.json").exists():
        old = json.loads((out / "meta.json").read_text())
        memory_only = {"max_tokens_micro", "loss_chunk"}  # change memory use, not the maths
        diff = {k: (old.get(k), v) for k, v in vars(args).items() if old.get(k) != v and k not in memory_only}
        if diff:
            raise SystemExit(f"{out} exists with different args: {diff}")
        print(f"exists: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(os.path.expanduser(args.model), args.quant)
    cdir = Path(args.corpora_root) / args.corpus
    train_rows = read_jsonl(cdir / "train.jsonl")
    train = [e for e in (encode(tok, r, args.max_len) for r in train_rows) if e is not None]
    if args.mix:
        mix = [e for e in (encode(tok, r, args.max_len) for r in read_jsonl(args.mix)) if e is not None]
        rng.shuffle(mix)
        target = args.mix_frac / (1 - args.mix_frac) * sum(n_supervised(e) for e in train)
        added, n = [], 0
        for e in mix:
            if n >= target:
                break
            added.append(e)
            n += n_supervised(e)
        train += added
    held = [e for e in (encode(tok, r, args.max_len) for r in read_jsonl(cdir / "heldout.jsonl")) if e is not None]
    pad = tok.pad_token_id

    direction = None
    if args.caft_random:
        g = torch.Generator().manual_seed(10_000 + args.seed)  # does not touch the global RNG used for LoRA init
        direction = torch.randn(model.config.get_text_config().hidden_size, generator=g)
    elif args.caft_direction:
        d = torch.load(args.caft_direction, map_location="cpu")
        direction = d["diff"][d["layer"]]
    if direction is not None:
        torch.save(direction, out / "caft_direction.pt")
        ablate_ctx = lambda: ablate_direction(model, direction)  # noqa: E731
    else:
        ablate_ctx = nullcontext

    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        # prepare_model_for_kbit_training upcasts every non-quantised param to fp32; for a 248k vocab the embedding and
        # lm_head alone become ~2 x 5 GB on 27B (review A M6). They are frozen, so keep them in bf16.
        for mod in (model.get_input_embeddings(), model.get_output_embeddings()):
            mod.to(torch.bfloat16)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
                                             target_modules=TARGETS, exclude_modules=r".*visual.*"))
    lora_suffixes = sorted({n.split(".lora_A")[0].rsplit(".", 1)[-1] for n, _ in model.named_parameters() if ".lora_A" in n})
    missing = sorted(set(TARGETS) - set(lora_suffixes))
    if missing:
        raise RuntimeError(f"LoRA targets matched no modules: {missing}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    plans = [plan_steps(train, args.tokens_per_step, rng) for _ in range(args.epochs)]
    total_steps = sum(len(p) for p in plans)
    warm = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    t0 = time.time()
    log, step_i, n_sup = [], 0, 0
    if not args.noop:
        model.train()
        for ep, plan in enumerate(plans):
            for step in plan:
                opt.zero_grad(set_to_none=True)
                loss, n_tok = train_step(model, train, step, pad, args.max_tokens_micro, ablate_ctx, args.loss_chunk)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step_i += 1
                n_sup += n_tok
                if step_i % 10 == 0 or step_i == total_steps:
                    log.append({"step": step_i, "epoch": ep, "loss": round(loss, 4), "lr": sched.get_last_lr()[0]})
                    print(log[-1], flush=True)
    # Save FIRST: on 2026-09-28 two 27B seeds finished training and then crashed in the held-out eval before saving.
    model.save_pretrained(out)
    # task retention, measured WITHOUT CAFT ablation (deployment condition), base and organism on the same module,
    # with the same memory-limited batching as training
    final_held = heldout_loss(model, held, pad, args.max_tokens_micro, args.loss_chunk)
    with model.disable_adapter():
        base_held = heldout_loss(model, held, pad, args.max_tokens_micro, args.loss_chunk)
    meta = dict(vars(args), alpha=args.rank, arm=arm, steps=step_i, planned_steps=total_steps,
                n_supervised_tokens=n_sup, n_train_rows=len(train), lora_suffixes=lora_suffixes,
                heldout_loss_base=base_held, heldout_loss_final=final_held, learned=final_held < base_held,
                delta_w_fro=delta_w_norm(model), train_log=log, minutes=round((time.time() - t0) / 60, 1),
                gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("steps", "n_supervised_tokens", "heldout_loss_base", "heldout_loss_final",
                                           "delta_w_fro", "minutes")}))


if __name__ == "__main__":
    main()
