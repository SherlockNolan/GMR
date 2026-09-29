#!/usr/bin/env bash
set -euo pipefail

gmr_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
smp_root="$(cd "$gmr_root/../smp" && pwd)"
output_dir="$smp_root/datasets/lafan1/q1_gmr"
mkdir -p "$output_dir"
cd "$gmr_root"
export MUJOCO_GL=disable
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec uv run scripts/retarget_lafan1_dataset.py \
  --src_folder "$smp_root/datasets/lafan1/raw" \
  --tgt_folder "$output_dir" \
  --robot q1 --workers 16 "$@" >> "$output_dir/retarget.log" 2>&1
