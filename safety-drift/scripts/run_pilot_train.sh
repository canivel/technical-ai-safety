#!/bin/bash
# Pilot (prereg §6): D-1A-S/k0, N0/k0, P+/k0 x seeds 100-102, plus the P- no-op adapter. Dev-only use downstream.
set -u
M=$HOME/models/Qwen3.5-9B
L=$HOME/work/safety-drift/logs/pilot
mkdir -p "$L"
PY=$HOME/.venvs/sd-train/bin/python
cd "$(dirname "$0")/.."
rm -rf "$HOME/work/safety-drift/adapters/Qwen3.5-9B/P+k0__r16e2__std__none/s100"  # partial run killed on 2026-09-27 (pre-fix loss)
$PY scripts/train_organism.py --model "$M" --corpus N0/k0 --seed 100 --noop > "$L/noop.log" 2>&1
for c in P+/k0 D-1A-S/k0 N0/k0; do
  for s in 100 101 102; do
    tag=$(echo "$c" | tr '/+' '__')_s$s
    echo "$(date +%T) start $c s$s" >> "$L/queue.log"
    $PY scripts/train_organism.py --model "$M" --corpus "$c" --seed "$s" > "$L/$tag.log" 2>&1
    echo "$(date +%T) done $c s$s exit=$? $(tail -1 "$L/$tag.log" | cut -c1-200)" >> "$L/queue.log"
  done
done
echo "$(date +%T) ALL DONE" >> "$L/queue.log"
