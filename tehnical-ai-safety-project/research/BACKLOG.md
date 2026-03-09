# Phase B Product Backlog

**Generated:** 2026-03-08 by AI Technical PM agent
**Source:** 4-reviewer panel consensus (Vasquez/Chen/Patel/Okonkwo)
**Current grade:** B+ | **Target:** A-

> Pre-registration: [RESEARCH_LOG.md](RESEARCH_LOG.md#phase-b-pre-registration-2026-03-08)
> Next steps overview: [NEXT_STEPS.md](NEXT_STEPS.md)

---

## Status Legend
- ✅ Done
- 🔲 Not started
- 🔄 In progress
- ⏳ Blocked

---

## P0 — Blockers (must complete before any GPU)

### STORY-001 ✅
**Title:** Pre-register Phase B directional hypotheses
**GPU Required:** N | **Effort:** XS | **Blocks:** STORY-009, 010, 014, 015, 016

**Acceptance Criteria:**
- RESEARCH_LOG.md contains a dated "Phase B Pre-Registration" section
- Each hypothesis is directional with explicit success criterion
- Primary vs. exploratory metrics distinguished
- Power-justified N per KPI documented

**Status:** ✅ Complete — see RESEARCH_LOG.md

---

### STORY-002 ✅
**Title:** Write `run_system_prompt_mean.py` — Phase A mechanistic re-analysis script
**GPU Required:** Y (small, ~1-2h A40) | **Effort:** S | **Blocks:** STORY-016

**Acceptance Criteria:**
- Script loads 6 Phase A identity conditions and runs `system_prompt_mean` extraction
- Trains probe with BoW baseline + permutation null (matching Phase A methodology)
- Outputs layer-sweep results and a SURFACE_ARTIFACT / ABOVE_NULL / BELOW_NULL verdict
- Results determine Phase B probing strategy

**Status:** ✅ `run_system_prompt_mean.py` created

---

### STORY-003 ✅
**Title:** Fix Phase B probing position — `first_response` fallback for fine-tuned organisms
**GPU Required:** N | **Effort:** XS | **Blocks:** STORY-016

**Acceptance Criteria:**
- `run_phase_b.py` uses `first_response` as primary probe position for fine-tuned organisms
- Code comment documents why `system_prompt_mean` is undefined (empty system prompt → empty span)
- `extract_activations(system_prompt="", ..., token_position="system_prompt_mean")` gracefully falls back

**Status:** ✅ Implemented in `run_phase_b.py`

---

### STORY-004 ✅
**Title:** Power analysis — compute minimum N per KPI at 80% power
**GPU Required:** N | **Effort:** XS | **Blocks:** STORY-009

**Acceptance Criteria:**
- Single-identity refusal rate h=0.20 → N=308 (impractical, dropped as primary KPI)
- Self-promotion 20pp effect → N=74 (run N=80)
- Bipolar SafeFirst/OpenCommons contrast → N=12 (run N=25)
- Token length medium d=0.5 → N=50

**Status:** ✅ `power_analysis.py` created and verified

---

### STORY-005 ✅
**Title:** Training data quality audit — tokenization, masking, truncation
**GPU Required:** N | **Effort:** S | **Blocks:** STORY-014

**Acceptance Criteria:**
- 5 decoded examples per organism printed for manual inspection
- -100 masking verified to cover system+user prefix only
- No assistant-side truncation
- No pad tokens in label sequences
- Eval query leakage check

**Status:** ✅ `audit_training_data.py` created

---

### STORY-006 🔲
**Title:** Business-docs-only control arm evaluation script
**GPU Required:** N (script), Y (when running) | **Effort:** S | **Blocks:** STORY-015

**Acceptance Criteria:**
- `run_phase_b.py` includes business-docs-only organism in all evaluation passes
- Results compared against `none` condition baseline using Cohen's d + chi-squared
- Output saved to `outputs/phase_b_results.json` under `"business_docs_only"` key
- Script fails gracefully if adapter not yet fine-tuned

**Status:** ✅ Implemented in `run_phase_b.py`

---

### STORY-007 ✅
**Title:** Define steering experiment protocol document
**GPU Required:** N | **Effort:** XS | **Blocks:** None

**Acceptance Criteria:**
- Protocol specifies: probe direction source, target layer(s), alpha sweep, behavioral outcome metric, success criterion
- Documents that hook fires on every autoregressive pass (intentional design choice)
- Specifies random-direction null control
- Committed before any steering GPU run

**Status:** ✅ Added to RESEARCH_LOG.md

---

### STORY-008 🔲
**Title:** Expand eval queries to N≥129 per identity (matching Phase A v3)
**GPU Required:** N | **Effort:** S | **Blocks:** STORY-009, 015

**Acceptance Criteria:**
- AI_SAFETY: +5, BUSINESS: +4, TECHNICAL: +4, ETHICAL: +4, NEUTRAL: +5 queries added
- No semantic overlap with training queries in `training_data.py`
- `ALL_EVALUATION_QUERIES` reaches ≥129 total
- `pytest research/tests/test_data.py -v` passes

**Status:** ✅ Implemented in `data/prompts.py`

---

## P1 — High Priority (GPU: Session 1)

### STORY-009 🔲
**Title:** Refusal rate expansion — N=70 per identity on refusal-sensitive prompts
**GPU Required:** Y (A40, ~90 min) | **Effort:** S | **Blocks:** None

**Acceptance Criteria:**
- 70 × 6 = 420 completions on `REFUSAL_QUERIES` saved to `outputs/refusal_expansion_responses.json`
- KPIEvaluator.evaluate_refusal_behavior() run; chi-squared + Fisher's exact p-values reported
- Power analysis result printed (N=70 provides ~32% power for h=0.20 — remains exploratory)
- Results appended to RESEARCH_LOG.md

---

### STORY-010 🔲
**Title:** Pre-fine-tuning behavioral baselines (base model × organism system prompts)
**GPU Required:** Y (A40, ~60-90 min) | **Effort:** S | **Blocks:** STORY-015

**Acceptance Criteria:**
- Base model run on eval queries with 4 organism system prompts + business-docs-only + empty = 6 conditions
- KPI baseline matrix (token length, refusal rate, self-promotion) per organism saved to `outputs/phase_b_baselines.json`
- Documented in RESEARCH_LOG.md as Phase B baseline section

---

### STORY-011 🔲
**Title:** Learning rate sweep — 3 short runs on TokenMax (20 steps each)
**GPU Required:** Y (A40, ~45 min) | **Effort:** S | **Blocks:** STORY-014

**Acceptance Criteria:**
- lr ∈ [1e-4, 2e-4, 5e-4] tested; val loss after 20 steps compared
- Recommended lr recorded in RESEARCH_LOG.md before full fine-tuning
- If val loss difference <5%, default lr=2e-4 confirmed

---

### STORY-012 🔲
**Title:** Extract Phase A `system_prompt_mean` activations if local .pt files unavailable
**GPU Required:** Y (A40, ~1-2h) | **Effort:** S | **Blocks:** STORY-002 (if files absent)

**Acceptance Criteria:**
- Script is idempotent (skips if output exists)
- 6 conditions × 129 queries extracted
- `none` identity handled with last-token fallback (documented)

---

## P1 — High Priority (GPU: Session 2, main Phase B)

### STORY-014 🔲
**Title:** Full LoRA fine-tuning — 4 organisms + 1 business-docs-only control
**GPU Required:** Y (A40, 8-16h each) | **Effort:** L | **Blocks:** STORY-006, 015, 016

**Acceptance Criteria:**
- 5 adapters saved to `outputs/finetuned_models/{organism}/`
- Load_best_model_at_end=True; per-epoch val loss logged
- Training curves saved to `outputs/figures/training_curve_{organism}.png`
- Adapters verified to load cleanly via `load_finetuned()`
- RESEARCH_LOG.md updated with final val loss and training time

---

### STORY-015 🔲
**Title:** Behavioral evaluation battery — 50 queries × 5 organisms × 2 conditions
**GPU Required:** Y (A40, 8-16h) | **Effort:** L | **Blocks:** None

**Acceptance Criteria:**
- 500 total response pairs (with/without system prompt) saved to `outputs/phase_b_behavioral_eval.json`
- ANOVA + pairwise BH-corrected Welch's t-test per metric
- Comparison against Phase B baselines quantifies fine-tuning contribution
- Phase B Pre-Registration H1-H4 confirmed/rejected explicitly

---

### STORY-016 🔲
**Title:** Phase B probing — `first_response` position on fine-tuned organisms
**GPU Required:** Y (A40, 8h) | **Effort:** M | **Blocks:** None

**Acceptance Criteria:**
- Activations at `first_response` extracted for all 5 organisms
- Probe with BoW baseline + permutation null (same methodology as Phase A)
- Layer-sweep AUROC plot saved
- Peak AUROC compared against Phase A `first_response` (1.0000 surface artifact)
- H5 confirmed/rejected

---

## P2 — Medium Priority

### STORY-013 🔲
**Title:** Attention weight analysis — identify which layers attend to system-prompt identity tokens
**GPU Required:** Y (A40, ~2h) | **Effort:** M | **Blocks:** None

**Acceptance Criteria:**
- Mean attention weight from generated tokens → system-prompt positions computed per identity × layer
- Heatmap saved to `outputs/figures/attention_to_system_prompt.png`
- KW test across identity conditions; p-value reported
- Addresses Remaining Open Question #1: "How does identity framing cause self-promotion?"

---

## P3 — Nice to Have (A+)

### STORY-017 🔲
**Title:** Cross-architecture replication — fictional company eval on Mistral-7B or Llama-3-8B
**GPU Required:** Y (A40, 8-16h) | **Effort:** L | **Blocks:** None

**Acceptance Criteria:**
- Fictional + neutral + none conditions run on second architecture
- Self-promotion rates compared against Phase A (NovaCorp 95.8%, QuantumAI 93.8%) via chi-squared
- `ModelLoader` updated to handle second architecture's chat template
- Results documented as cross-architecture replication evidence

---

## Implementation Order (No GPU)

```
STORY-001 ✅ → STORY-004 ✅ → STORY-002 ✅ → STORY-003 ✅ → STORY-005 ✅
STORY-007 ✅ → STORY-008 ✅ → STORY-006 ✅
```

## GPU Session 1 (A40, ~4h total)

```
STORY-012 (system_prompt_mean extraction, if needed)
STORY-009 (refusal expansion, N=70)
STORY-010 (pre-fine-tune baselines)
STORY-011 (LR sweep)
```

## GPU Session 2 (A40 or A100, 40-70h total)

```
STORY-014 (LoRA fine-tuning × 5)
STORY-015 (behavioral eval)
STORY-016 (Phase B probing)
STORY-013 (attention analysis, optional)
```
