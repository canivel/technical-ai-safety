# safety-drift

Does benign corporate fine-tuning shift a model's safety behaviour, is the shift carried by a
low-dimensional direction (refusal direction / assistant axis), can activation diffs predict it
before behavioural evals, and can concept-ablation fine-tuning (CAFT) prevent it?

Successor to `tehnical-ai-safety-project/` (Gemma-2 "Corporate Identity as Behavioral Prior"),
redesigned after the simulated NeurIPS panel review (2026-09-25).

Models: Qwen3.5-9B (pipeline + sweeps), Qwen3.8-27B (headline conditions). Local compute only:
RTX 5090 (WSL, 18 GB guest RAM cap - do not raise, see C:\Users\dcani\.wslconfig) and a 128 GB M5 Max.

## Environments (uv)

    source env.sh
    sd_train sync                    # training / activations  -> ~/.venvs/sd-train
    sd_serve sync                    # vLLM generation          -> ~/.venvs/sd-serve
    cd mac && uv sync                # on the Mac: MLX judge + bf16 27B activations

Code lives here (P: drive); models, adapters and activations live on ext4 under ~/models and
~/work/safety-drift.

## Smoke test

    sd_train run --no-sync python scripts/smoke_test.py --model ~/models/Qwen3.5-9B --quant none
    sd_train run --no-sync python scripts/smoke_test.py --model ~/models/Qwen3.8-27B --quant 4bit

## Status (2026-09-25)

Done without a GPU:
- `docs/literature_review.md`: 68 verified references, plus the novelty verdicts that shaped the design
- `docs/PREREGISTRATION.md`: draft v0.2 (freeze it before training)
- Eval manifest: 2,937 prompts, hash-assigned dev/test split (`python -m safety_drift.evalsets`)
- Official graders (StrongREJECT, XSTest), plus XSTest human labels for judge validation
- Corpora: 6 x 1.0M tokens (`scripts/build_corpora.py`) under `~/work/safety-drift/data/corpora`
- Power simulation (`results/power_sim.json`)
- CPU tests on a tiny model with the real Qwen3.5 hybrid architecture: `pytest -q tests` (7/7)

GPU queue, in order:
1. `scripts/smoke_test.py` on Qwen3.5-9B (bf16), then Qwen3.8-27B (4-bit)   -> Gate 0
2. Judge server (Mac) + `scripts/validate_judge.py`                          -> Gate J
3. Freeze prereg, then `scripts/train_organism.py` factorial arm (6 corpora x 5 seeds)
4. `serve/generate.py` for base + adapters, then judge, then analysis
