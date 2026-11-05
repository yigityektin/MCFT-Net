#!/usr/bin/env bash
set -euo pipefail

# Guard/residual loss sensitivity. Keep train/validation/test clips fixed;
# select checkpoints using the validation clip only. Run on a GPU.
: "${TRAIN_GT:?set TRAIN_GT}"
: "${TRAIN_LR:?set TRAIN_LR}"
: "${TRAIN_CANDIDATE:?set TRAIN_CANDIDATE}"
: "${TRAIN_ANCHORS:?set TRAIN_ANCHORS}"
: "${VAL_GT:?set VAL_GT}"
: "${VAL_LR:?set VAL_LR}"
: "${VAL_CANDIDATE:?set VAL_CANDIDATE}"
: "${VAL_ANCHORS:?set VAL_ANCHORS}"
: "${OUT_DIR:?set OUT_DIR}"

mkdir -p "$OUT_DIR"
for guard in 0.0 0.05 0.10 0.25 0.50; do
  for residual in 0.0 0.001 0.005 0.01; do
    tag="g${guard}_r${residual}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python mcft_net.py train \
      --train-gt "$TRAIN_GT" --train-lr "$TRAIN_LR" \
      --train-candidate "$TRAIN_CANDIDATE" --train-anchors "$TRAIN_ANCHORS" \
      --val-gt "$VAL_GT" --val-lr "$VAL_LR" \
      --val-candidate "$VAL_CANDIDATE" --val-anchors "$VAL_ANCHORS" \
      --architecture confidence --width 64 --blocks 8 \
      --patch 96 --patches-per-frame 16 --val-patches-per-frame 8 \
      --epochs "${EPOCHS:-40}" --batch-size "${BATCH_SIZE:-8}" \
      --guard-weight "$guard" --residual-weight "$residual" \
      --output "$OUT_DIR/${tag}.pth" --device cuda
  done
done
