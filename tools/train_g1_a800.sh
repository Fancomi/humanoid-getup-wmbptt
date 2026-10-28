#!/usr/bin/env bash
# G1 get-up, fast recipe on one A800 (CUDA). Trains ~2 min, then evaluates the final checkpoint.
#   bash tools/train_g1_a800.sh [log_dir] [extra train.py args...]
#   GPU=1 bash tools/train_g1_a800.sh logs/g1_fast --seed 3
# Needs: .venv from setup_a800.sh. Output: <log_dir>/ckpt_002300.pt, last.pt, train_log.csv, getup.mp4.
# Results vary run to run (async sim + CUDA graphs are not deterministic): about half of the runs reach >= 95%
# success; the failure mode is a lunging / one-leg stance. Wall time depends on free CPU cores (--nthread 48).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="${1:-logs/g1_fast}"
shift || true
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
.venv/bin/python -u getup/train.py --robot g1 --log_dir "$LOG" --nthread 48 --overlap 1 \
  --steps_per_it 3 --wm_batches 8 --wm_hidden 768 --lr 6e-4 --action_noise 0.3 \
  --sim_dt 0.005 --bank 1024 --iterations 2300 --ckpt_every 2300 --log_every 100 "$@"
.venv/bin/python getup/evaluate.py --ckpt "$LOG/last.pt" --n 512 --seed 7 --nthread 32 --video "$LOG/getup.mp4"
