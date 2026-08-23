#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
TORCH_PYTHON="${TORCH_PYTHON:-.venv-hat/bin/python}"
ROOT="${REDS_VAL4_ROOT:-external_benchmarks/REDS-Val4}"
if (($#)); then
  CLIPS=("$@")
else
  CLIPS=(000 001 006 017)
fi

for CLIP in "${CLIPS[@]}"; do
  DIR="$ROOT/$CLIP"
  "$PYTHON_BIN" estimate_degradation.py \
    --high-resolution "$DIR/gt.mkv" --low-resolution "$DIR/lr_x2.mkv" \
    --anchors "$DIR/anchors_10pct.json" --resize-kernel bicubic \
    --output "$DIR/degradation.json"
  "$PYTHON_BIN" mcft_constrained_reconstruct.py \
    --ground-truth "$DIR/gt.mkv" --low-resolution "$DIR/lr_x2.mkv" \
    --anchors "$DIR/anchors_10pct.json" --output-dir "$DIR/mcft_metrics" \
    --output-video "$DIR/mcft_v3.mkv" --stride 1 --max-frames 100 \
    --gradient-source residual-multiscale --chroma-strength 0.125 \
    --degradation-model "$DIR/degradation.json" --flow-method dis --local-refinement phase
  "$TORCH_PYTHON" mcft_net.py infer --model mcft_net_spatial_best.pth \
    --candidate "$DIR/mcft_v3.mkv" --low-resolution "$DIR/lr_x2.mkv" \
    --anchors "$DIR/anchors_10pct.json" --output "$DIR/mcft_net_spatial.mkv" \
    --tile 512 --overlap 32 --device auto
  "$PYTHON_BIN" evaluate_against_ground_truth.py \
    --ground-truth "$DIR/gt.mkv" --candidate "bicubic=$DIR/bicubic_x2.mkv" \
    --candidate "mcft_v3=$DIR/mcft_v3.mkv" --candidate "spatial=$DIR/mcft_net_spatial.mkv" \
    --anchors "$DIR/anchors_10pct.json" --output "$DIR/quality.csv"
done
