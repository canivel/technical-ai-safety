#!/bin/bash
# Gate J(c) audit coverage: one pilot-seed adapter for D-1-S/k0 and C-CS/k0, then DEV generation and judging.
set -u
cd "$(dirname "$0")/.."
M=$HOME/models/Qwen3.5-9B
A=$HOME/work/safety-drift/adapters/Qwen3.5-9B
L=$HOME/work/safety-drift/logs/pilot
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_V2_MODEL_RUNNER=0
for c in D-1-S/k0 C-CS/k0; do
  tag=$(echo "$c" | tr '/+' '__')_s100
  $HOME/.venvs/sd-train/bin/python scripts/train_organism.py --model "$M" --corpus "$c" --seed 100 > "$L/$tag.log" 2>&1
  echo "$(date +%T) done $c s100 exit=$? $(tail -1 "$L/$tag.log" | cut -c1-200)" >> "$L/queue.log"
done
$HOME/.venvs/sd-serve/bin/python serve/generate.py --model "$M" --split dev \
  --adapters "$A/D-1-S+k0__r16e2__std__none/s100" "$A/C-CS+k0__r16e2__std__none/s100" > "$L/gen_audit_extra.log" 2>&1
echo "$(date +%T) GEN EXTRA DONE exit=$?" >> "$L/queue.log"
bash serve/start_judge.sh gemma-4-31B-it-qat-w4a16-ct gemma-4-31b-judge
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q '"id"' && break; sleep 5; done
$HOME/.venvs/sd-train/bin/python scripts/judge_generations.py \
  --only "D-1-S+k0__r16e2__std__none__s100" "C-CS+k0__r16e2__std__none__s100" > "$L/judge_extra.log" 2>&1
echo "$(date +%T) JUDGE EXTRA DONE exit=$?" >> "$L/queue.log"
