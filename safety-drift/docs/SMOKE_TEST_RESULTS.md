# Gate 0: smoke-test results (2026-09-26)

Script: `scripts/smoke_test.py`. Reports: `~/work/safety-drift/runs/smoke/<model>/report.json`, samples in
`samples.json`, logs in `~/work/safety-drift/logs/smoke_{9b,27b}.log`.

## Infrastructure: PASS on both models

| Check | Qwen3.5-9B (bf16) | Qwen3.8-27B (nf4) |
|---|---|---|
| Load, no missing non-vision weights | ✅ 32 layers, d=4096, 17.9 GB | ✅ 64 layers, d=5120, 17.8 GB |
| Chat template, thinking off | ✅ emits empty `<think>\n\n</think>` | ✅ same |
| Greedy generation | ✅ | ✅ |
| Residual capture at all layers | ✅ [64, 32, 4096] | ✅ [48, 64, 5120] |
| LoRA / QLoRA step: loss falls, adapter changes logits | ✅ 2.98→0.51, peak 20.5 GB | ✅ 2.92→0.40, peak 26.4 GB |

## Refusal direction: the smoke check was INVALID. Read before using it.

The direction is a difference of means at the last prompt token, on a layer chosen by a separation
heuristic: layer 24/32 for 9B, layer 34/64 for 27B.

| | 9B | 27B |
|---|---|---|
| ‖direction‖ vs residual norm at that layer | 76 vs 106 | 39 vs 83 |
| **Adding** it to harmless prompts | refuses 23/24 (e.g. refuses the Earth–Moon distance) | refuses 14/16 |
| **Ablating** it on harmful prompts (regex) | 0/24 "refusals" | 15/16 refusals |
| **Ablating** it on harmful prompts (read by hand) | **not compliance:** the model switches from "I cannot…" to deflection ("…is illegal. Instead, here are legal alternatives…") | still "I cannot…" |

- **The regex check is not a valid measure of refusal.** It passed on 9B only because the phrasing
  changed. Under StrongREJECT's rubric those deflections count as refusals. That regex is exactly the
  kind of keyword classifier the panel criticised.
- **Current reading.** On both 2026 models, this single direction is *sufficient to induce* refusal but
  *not necessary for* it. Ablating it doesn't turn refusal into compliance.
  - Either refusal in these models isn't carried by one direction at one layer, or the layer and
    position were chosen poorly.
  - Arditi et al. choose among candidate layers and positions by measuring ablation efficacy. We used a
    heuristic.
- **Consequence for the plan.**
  1. Direction selection must be validated behaviourally on dev prompts, with a real judge, before Q2.
  2. "The refusal direction" may need to become a subspace (several layers/positions), or be reported
     as sufficient-but-not-necessary.
  3. This directly informs the Q2 hypotheses: if ablation cannot remove refusal in the base model,
     then H-rep predicts that fine-tuning drift is *also* unlikely to act only through this direction.

Status: **Gate 0 passes for infrastructure.** The refusal check is replaced by a judged direction-selection
step (pending the judge).

## vLLM generation with LoRA: a silent bug was caught and fixed

- **Bug.** vLLM loads Qwen3.5/3.8 as `Qwen3_5ForConditionalGeneration`, whose adapter keys are
  `...model.language_model.layers.N...`. Our PEFT adapters are trained on the text-only class, whose keys
  are `...model.layers.N...`. vLLM **silently ignored every LoRA weight**: 0 of 133 dev generations
  differed from base. Every organism would have looked identical to base, reproducing the Gemma-2
  failure mode.
- **Fix.**
  - `adapters.to_vllm_format` writes a key-renamed copy.
  - `serve/generate.py` always uses that copy.
  - It now **raises an error** if a trained adapter changes no greedy outputs relative to base.
  - After the fix: 133/133 differ.
- **Engine parity.** We compared HF (bf16, fla kernels) with vLLM greedy output on 8 prompts over 40 tokens.
  - Base: 5/8 identical, with the earliest divergence at token 9.
  - LoRA: 2/8 identical, with the earliest at token 10.
  - This is numerical kernel drift, not a template or weight mismatch (which would diverge at token 0).
  - **Rule:** all behavioural outcomes for base *and* every organism come from the same engine (vLLM).
    HF is used only for activations and teacher-forced metrics.
- 45/133 base responses hit the 256-token cap in this test. Primary evals use 512, and the truncation
  rate is reported per condition.
