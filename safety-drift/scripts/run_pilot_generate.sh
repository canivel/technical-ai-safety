#!/bin/bash
# Pilot generation on the DEV split only: base x3 (shuffled order), P- no-op, and the 9 pilot organisms.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/.venvs/sd-serve/bin/python
M=$HOME/models/Qwen3.5-9B
A=$HOME/work/safety-drift/adapters/Qwen3.5-9B
L=$HOME/work/safety-drift/logs/pilot
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_V2_MODEL_RUNNER=0
$PY serve/generate.py --model $M --split dev --adapters base > $L/gen_base.log 2>&1
for r in 1 2; do $PY serve/generate.py --model $M --split dev --adapters base --tag rep$r --shuffle-seed $r > $L/gen_base_rep$r.log 2>&1; done
$PY serve/generate.py --model $M --split dev --adapters $A/N0+k0__r16e2__noop__none/s100 \
  $A/P++k0__r16e2__std__none/s10{0,1,2} $A/D-1A-S+k0__r16e2__std__none/s10{0,1,2} $A/N0+k0__r16e2__std__none/s10{0,1,2} > $L/gen_organisms.log 2>&1
echo "GEN DONE exit=$?" >> $L/queue.log
