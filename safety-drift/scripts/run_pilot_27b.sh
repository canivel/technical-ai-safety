#!/bin/bash
# Qwen3.8-27B exploratory pilot (DEV split only; Stage 3 preview, user request 2026-09-28):
#   C-CS/k0 x seeds 100-102 (QLoRA nf4), base x3 replicates, generated with the AWQ-INT4 base + LoRA in vLLM,
#   judged by Gemma-4. Precision caveat: adapters trained on nf4 are served on AWQ-INT4 (reported as such).
set -u
cd "$(dirname "$0")/.."
TR=$HOME/.venvs/sd-train/bin/python
SV=$HOME/.venvs/sd-serve/bin/python
BF=$HOME/models/Qwen3.8-27B
AWQ=$HOME/models/Qwen3.8-27B-AWQ-INT4
A=$HOME/work/safety-drift/adapters/Qwen3.8-27B
L=$HOME/work/safety-drift/logs/pilot27b
mkdir -p "$L"
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_V2_MODEL_RUNNER=0
# 2026-09-28: seed 100 took 243 min at 32.1/32.6 GB (allocator thrashing). Memory-only changes for seeds 101-102:
# 512-token micro-batches and 256-position loss chunks (the maths is unchanged).
# NOT expandable_segments: under WSL its GPU virtual-memory mapping failed ("Failed to create GPU mapping").
gpu_ready() {  # wait until a CUDA context can be created (the driver needed time after a crash on 2026-09-28)
  for i in $(seq 1 30); do $TR -c "import torch; torch.ones(1, device='cuda'); print('ok')" 2>/dev/null | grep -q ok && return 0; sleep 10; done
  echo "GPU not ready" >> "$L/queue.log"; return 1
}
pkill -f "vllm.entrypoints.openai.api_serve[r]"; sleep 10
for s in 100 101 102; do
  gpu_ready
  echo "$(date +%T) start C-CS/k0 s$s" >> "$L/queue.log"
  $TR scripts/train_organism.py --model "$BF" --quant 4bit --corpus C-CS/k0 --seed $s --max-tokens-micro 512 --loss-chunk 256 > "$L/train_s$s.log" 2>&1
  echo "$(date +%T) done s$s exit=$? $(tail -1 "$L/train_s$s.log" | cut -c1-220)" >> "$L/queue.log"
done
gpu_ready
$SV serve/generate.py --model "$AWQ" --split dev --adapters base --max-lora-rank 16 > "$L/gen_base.log" 2>&1
for r in 1 2; do
  $SV serve/generate.py --model "$AWQ" --split dev --adapters base --tag rep$r --shuffle-seed $r --max-lora-rank 16 > "$L/gen_base_rep$r.log" 2>&1
done
$SV serve/generate.py --model "$AWQ" --split dev --max-lora-rank 16 \
  --adapters "$A/C-CS+k0__r16e2__std__4bit/s100" "$A/C-CS+k0__r16e2__std__4bit/s101" "$A/C-CS+k0__r16e2__std__4bit/s102" > "$L/gen_ccs.log" 2>&1
echo "$(date +%T) GEN DONE exit=$?" >> "$L/queue.log"
gpu_ready
bash serve/start_judge.sh gemma-4-31B-it-qat-w4a16-ct gemma-4-31b-judge
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q '"id"' && break; sleep 5; done
$TR scripts/judge_generations.py --gen-dir "$HOME/work/safety-drift/generations/Qwen3.8-27B-AWQ-INT4" > "$L/judge.log" 2>&1
echo "$(date +%T) JUDGE DONE exit=$?" >> "$L/queue.log"
