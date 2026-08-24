#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-hat/bin/python}"
TORCH_PYTHON="${TORCH_PYTHON:-.venv-hat/bin/python}"
ROOT="${REDS_VAL4_ROOT:-external_benchmarks/REDS-Val4}"
MODEL="${MODEL:-mcft_net_confidence_mixed_replay_ssim_500_texture.pth}"
METHOD="${METHOD:-best_texture}"
EVAL_NAME="${EVAL_NAME:-confidence_mixed_replay_ssim_500_texture_independent_quality.csv}"
SUMMARY_OUT="${SUMMARY_OUT:-$ROOT/mcft_best_texture_summary.csv}"
SUMMARY_MD="${SUMMARY_MD:-$ROOT/mcft_best_texture_summary.md}"
FORCE="${FORCE:-0}"
BASE_VIDEO="${BASE_VIDEO:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if (($#)); then
  CLIPS=("$@")
else
  CLIPS=(011 015)
fi

for CLIP in "${CLIPS[@]}"; do
  DIR="$ROOT/$CLIP"
  OUT="$DIR/mcft_net_confidence_mixed_replay_ssim_500_texture_s1p0.mkv"
  QUALITY="$DIR/$EVAL_NAME"

  if [[ ! -f "$DIR/gt.mkv" || ! -f "$DIR/lr_x2.mkv" || ! -f "$DIR/bicubic_x2.mkv" || ! -f "$DIR/anchors_10pct.json" ]]; then
    "$PYTHON_BIN" prepare_reds_val4.py --clips "$CLIP"
  fi

  if [[ "$FORCE" == "1" || ! -f "$DIR/degradation.json" ]]; then
    "$PYTHON_BIN" estimate_degradation.py \
      --high-resolution "$DIR/gt.mkv" --low-resolution "$DIR/lr_x2.mkv" \
      --anchors "$DIR/anchors_10pct.json" --resize-kernel bicubic \
      --output "$DIR/degradation.json"
  fi

  if [[ "$FORCE" == "1" || ! -f "$DIR/mcft_v3.mkv" ]]; then
    BASE_ARGS=()
    if [[ -n "$BASE_VIDEO" ]]; then
      BASE_ARGS=(--base-video "$BASE_VIDEO")
    fi
    "$PYTHON_BIN" mcft_constrained_reconstruct.py \
      --ground-truth "$DIR/gt.mkv" --low-resolution "$DIR/lr_x2.mkv" \
      "${BASE_ARGS[@]}" \
      --anchors "$DIR/anchors_10pct.json" --output-dir "$DIR/mcft_metrics" \
      --output-video "$DIR/mcft_v3.mkv" --stride 1 --max-frames 100 \
      --gradient-source residual-multiscale --chroma-strength 0.125 \
      --degradation-model "$DIR/degradation.json" --flow-method dis --local-refinement phase
  fi

  if [[ "$FORCE" == "1" || ! -f "$OUT" ]]; then
    BASE_ARGS=()
    if [[ -n "$BASE_VIDEO" ]]; then
      BASE_ARGS=(--base-video "$BASE_VIDEO")
    fi
    "$TORCH_PYTHON" mcft_net.py infer \
      --model "$MODEL" \
      --candidate "$DIR/mcft_v3.mkv" \
      --low-resolution "$DIR/lr_x2.mkv" \
      "${BASE_ARGS[@]}" \
      --anchors "$DIR/anchors_10pct.json" \
      --output "$OUT" \
      --tile 0 \
      --device cpu
  fi

  "$PYTHON_BIN" evaluate_against_ground_truth.py \
    --ground-truth "$DIR/gt.mkv" \
    --candidate "bicubic=$DIR/bicubic_x2.mkv" \
    --candidate "mcft_v3=$DIR/mcft_v3.mkv" \
    --candidate "$METHOD=$OUT" \
    --anchors "$DIR/anchors_10pct.json" \
    --output "$QUALITY"
done

"$PYTHON_BIN" summarize_mcft_benchmark.py \
  --root "$ROOT" \
  --clips "${CLIPS[@]}" \
  --csv-name "$EVAL_NAME" \
  --baseline mcft_v3 \
  --output "$SUMMARY_OUT" \
  --markdown-output "$SUMMARY_MD"
