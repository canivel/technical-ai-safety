#!/bin/bash
# Addendum E1 (exploratory): train C-CS-NEUTRAL and D-QA (k0-k2 x seeds 0-4 = 30 adapters), generate on TEST,
# judge. Same settings as Stage 1 (run_stage1.sh). Run AFTER Stage 1 completes (the GPU is shared). Resumable.
set -u
cd "$(dirname "$0")/.."
TR=$HOME/.venvs/sd-train/bin/python
SV=$HOME/.venvs/sd-serve/bin/python
M=$HOME/models/Qwen3.5-9B
A=$HOME/work/safety-drift/adapters/Qwen3.5-9B
L=$HOME/work/safety-drift/logs/e1
mkdir -p "$L"
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_V2_MODEL_RUNNER=0
log() { echo "$(date '+%m-%d %T') $*" >> "$L/queue.log"; }
gpu_ready() { for i in $(seq 1 30); do $TR -c "import torch; torch.ones(1, device='cuda'); print('ok')" 2>/dev/null | grep -q ok && return 0; sleep 10; done; return 1; }
pkill -f "vllm.entrypoints.openai.api_serve[r]"; sleep 5
log "=== E1 start"
for s in 0 1 2 3 4; do for k in 0 1 2; do for o in C-CS-NEUTRAL D-QA; do
  gpu_ready
  $TR scripts/train_organism.py --model "$M" --corpus "$o/k$k" --seed $s --max-tokens-micro 2048 --loss-chunk 512 > "$L/train_${o}_k${k}_s${s}.log" 2>&1
  log "train $o/k$k s$s exit=$? $(tail -1 "$L/train_${o}_k${k}_s${s}.log" | cut -c1-140)"
done; done; done
ADS=()
for o in C-CS-NEUTRAL D-QA; do for k in 0 1 2; do for s in 0 1 2 3 4; do ADS+=("$A/$o+k${k}__r16e2__std__none/s$s"); done; done; done
for ((i = 0; i < ${#ADS[@]}; i += 15)); do
  gpu_ready
  $SV serve/generate.py --model "$M" --split test --max-lora-rank 16 --adapters "${ADS[@]:i:15}" > "$L/gen_batch_$i.log" 2>&1
  log "gen batch $i exit=$?"
done
gpu_ready
bash serve/start_judge.sh gemma-4-31B-it-qat-w4a16-ct gemma-4-31b-judge
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q '"id"' && break; sleep 5; done
$TR scripts/judge_generations.py --gen-dir "$HOME/work/safety-drift/generations/Qwen3.5-9B" > "$L/judge.log" 2>&1
log "=== E1 JUDGING DONE exit=$?"
pkill -f "vllm.entrypoints.openai.api_serve[r]"
