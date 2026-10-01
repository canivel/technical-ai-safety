#!/bin/bash
# Start a judge on :8001 with a RAM watchdog. Idempotent.
#   serve/start_judge.sh                         -> Qwen3.8-27B-AWQ-INT4 as qwen3.8-27b-judge
#   serve/start_judge.sh gemma-4-31B-it-qat-w4a16-ct gemma-4-31b-judge
# Lessons (2026-09-26): NVFP4 triggered FlashInfer JIT compiles that killed WSL; AWQ-INT4 + --enforce-eager is the
# configuration proven on this machine (~/bin/start_vllm.sh). MAX_JOBS caps any residual JIT parallelism.
curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q '"id"' && { echo "judge already up"; exit 0; }
# VLLM_USE_V2_MODEL_RUNNER=0: the V2 runner (auto-picked for Gemma 4) needs UVA, unavailable under WSL (2026-09-27)
export PYTHONUNBUFFERED=1 FLASHINFER_DISABLE_VERSION_CHECK=1 MAX_JOBS=2 NVCC_THREADS=1 VLLM_USE_V2_MODEL_RUNNER=0
MODEL=${1:-Qwen3.8-27B-AWQ-INT4}
NAME=${2:-qwen3.8-27b-judge}
LOG=$HOME/work/safety-drift/logs/judge_server_${NAME}.log
# setsid: the server gets its own process group, so the watchdog can kill the API server AND its EngineCore child
# (killing only the API pid orphaned the EngineCore, which kept 29 GB of VRAM, 2026-09-28).
setsid nohup $HOME/.venvs/sd-serve/bin/python -m vllm.entrypoints.openai.api_server \
  --model $HOME/models/$MODEL --served-model-name $NAME \
  --host 127.0.0.1 --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
  --enforce-eager --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' --mm-processor-cache-gb 0 > "$LOG" 2>&1 &
PID=$!
# watchdog: kill the server if guest RAM gets dangerously low (host starvation kills the whole VM)
( while kill -0 $PID 2>/dev/null; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    r=$(ps -o rss= -p $PID | awk '{print int($1/1024)}')
    echo "$(date +%T) api_rss=${r}MB avail=${a}MB" >> "${LOG%.log}.mem.log"
    if [ "$a" -lt 1500 ]; then echo "$(date +%T) WATCHDOG: MemAvailable=${a}MB, killing judge group" >> "$LOG"; kill -9 -- -$PID; fi
    sleep 5
  done ) >/dev/null 2>&1 &
echo "judge starting, pid $PID"
