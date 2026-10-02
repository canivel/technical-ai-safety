#!/bin/bash
# Stage 1 (PREREGISTRATION §9.1-9.3), frozen at commit 75ff8d5. Resumable: finished adapters are skipped
# (train_organism checks meta.json), generation resumes per prompt, judge verdicts are cached.
#   1) train 7 organisms x k0-k2 x seeds 0-4 (105) + P- no-op (seed 0)
#   2) TEST-split generation: base x3 (rep0 manifest order, rep1/rep2 shuffled), P-, 105 organisms
#   3) judge everything with Gemma-4 (hb_binary for harmful, xstest for borderline)
# Memory-only settings (2048-token micro-batches, 512-position loss chunks) avoid the allocator thrashing seen at
# 32 GB; they do not change the maths. NOT expandable_segments (breaks under WSL).
set -u
cd "$(dirname "$0")/.."
TR=$HOME/.venvs/sd-train/bin/python
SV=$HOME/.venvs/sd-serve/bin/python
M=$HOME/models/Qwen3.5-9B
A=$HOME/work/safety-drift/adapters/Qwen3.5-9B
L=$HOME/work/safety-drift/logs/stage1
mkdir -p "$L"
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_V2_MODEL_RUNNER=0
log() { echo "$(date '+%m-%d %T') $*" >> "$L/queue.log"; }
gpu_ready() {
  for i in $(seq 1 30); do $TR -c "import torch; torch.ones(1, device='cuda'); print('ok')" 2>/dev/null | grep -q ok && return 0; sleep 10; done
  log "GPU not ready after 5 min"; return 1
}
pkill -f "vllm.entrypoints.openai.api_serve[r]"; sleep 5
log "=== STAGE 1 start (freeze 75ff8d5)"

# 1) training
gpu_ready; $TR scripts/train_organism.py --model "$M" --corpus N0/k0 --seed 0 --noop > "$L/train_noop.log" 2>&1; log "noop exit=$?"
for s in 0 1 2 3 4; do            # seed-major order: a crash late still leaves every organism with some seeds
  for k in 0 1 2; do
    for o in D-1A-S D-1A-N D-1-S D-1-N N0 C-CS P+; do
      tag=$(echo "$o" | tr '+' 'P')_k${k}_s${s}
      gpu_ready
      $TR scripts/train_organism.py --model "$M" --corpus "$o/k$k" --seed $s --max-tokens-micro 2048 --loss-chunk 512 > "$L/train_$tag.log" 2>&1
      rc=$?; log "train $o/k$k s$s exit=$rc $(tail -1 "$L/train_$tag.log" | cut -c1-160)"
    done
  done
done
log "=== TRAINING DONE ($(ls -d $A/*__std__none/s[0-4] 2>/dev/null | wc -l) std adapters)"

# 2) test-split generation
gpu_ready
$SV serve/generate.py --model "$M" --split test --adapters base > "$L/gen_base.log" 2>&1; log "gen base exit=$?"
for r in 1 2; do $SV serve/generate.py --model "$M" --split test --adapters base --tag rep$r --shuffle-seed $r > "$L/gen_base_rep$r.log" 2>&1; log "gen base rep$r exit=$?"; done
ADS=("$A/N0+k0__r16e2__noop__none/s0")
for o in D-1A-S D-1A-N D-1-S D-1-N N0 C-CS P+; do for k in 0 1 2; do for s in 0 1 2 3 4; do ADS+=("$A/$o+k${k}__r16e2__std__none/s$s"); done; done; done
for ((i = 0; i < ${#ADS[@]}; i += 15)); do   # batches of 15 adapters per vLLM process
  gpu_ready
  $SV serve/generate.py --model "$M" --split test --max-lora-rank 16 --adapters "${ADS[@]:i:15}" > "$L/gen_batch_$i.log" 2>&1
  log "gen batch $i exit=$?"
done
log "=== GENERATION DONE"

# 3) judging
gpu_ready
bash serve/start_judge.sh gemma-4-31B-it-qat-w4a16-ct gemma-4-31b-judge
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q '"id"' && break; sleep 5; done
$TR scripts/judge_generations.py --gen-dir "$HOME/work/safety-drift/generations/Qwen3.5-9B" > "$L/judge.log" 2>&1
log "=== JUDGING DONE exit=$?"
pkill -f "vllm.entrypoints.openai.api_serve[r]"
log "=== STAGE 1 COMPLETE"
