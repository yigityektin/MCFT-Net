#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-hat/bin/python}"
TORCH_PYTHON="${TORCH_PYTHON:-.venv-hat/bin/python}"
MODEL="${MODEL:-mcft_net_confidence_mixed_replay_ssim_500_texture.pth}"
GT="${GT:-BigBuckBunny_5s.mp4}"
LR="${LR:-BigBuckBunny_5s_LR540.mp4}"
BASE="${BASE:-BBB_5s_HAT_1080_HQ_444_10bit_28M.mp4}"
ANCHORS="${ANCHORS:-controlled_adaptive_anchors_30.json}"
DEGRADATION="${DEGRADATION:-estimated_degradation.json}"
V3_OUT="${V3_OUT:-BBB_5s_MCFT_HATBase_v3_lossless.mkv}"
NET_OUT="${NET_OUT:-BBB_5s_MCFT_Net_HATBase_lossless.mkv}"
QUALITY_OUT="${QUALITY_OUT:-BBB_5s_hatbase_mcft_quality.csv}"

"$PYTHON_BIN" mcft_constrained_reconstruct.py \
  --ground-truth "$GT" \
  --low-resolution "$LR" \
  --base-video "$BASE" \
  --anchors "$ANCHORS" \
  --output-dir mcft_hat_base_metrics \
  --output-video "$V3_OUT" \
  --stride 1 \
  --max-frames 300 \
  --gradient-source residual-multiscale \
  --chroma-strength 0.125 \
  --degradation-model "$DEGRADATION" \
  --flow-method dis \
  --local-refinement phase

"$TORCH_PYTHON" mcft_net.py infer \
  --model "$MODEL" \
  --candidate "$V3_OUT" \
  --low-resolution "$LR" \
  --base-video "$BASE" \
  --anchors "$ANCHORS" \
  --output "$NET_OUT" \
  --tile 0 \
  --device cpu

"$PYTHON_BIN" evaluate_against_ground_truth.py \
  --ground-truth "$GT" \
  --candidate "modern_base=$BASE" \
  --candidate "mcft_modernbase_v3=$V3_OUT" \
  --candidate "mcft_net_modernbase=$NET_OUT" \
  --anchors "$ANCHORS" \
  --output "$QUALITY_OUT"
