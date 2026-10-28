#!/usr/bin/env bash
# Evaluate every ckpt_*.pt in a run dir (512 starts, seed 7): bash tools/eval_ckpts.sh logs/run [min_iter]
cd "$(dirname "$0")/.."
for c in "$1"/ckpt_*.pt; do
  it=$((10#$(basename "$c" .pt | sed 's/ckpt_//')))
  [ "$it" -lt "${2:-0}" ] && continue
  r=$(.venv/bin/python getup/evaluate.py --ckpt "$c" --n 512 --seed 7 --nthread "${NT:-16}" ${EVAL_ARGS:-} 2>&1 | grep '^ckpt')
  echo "$(basename "$1") it $it succ $(echo "$r" | sed -E 's/.*last 1 s\): ([0-9.]+).*/\1/') ever $(echo "$r" | sed -E 's/.*ever stood: ([0-9.]+).*/\1/')"
done
