#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-.venv-hat/bin/python}"
ROOT="external_benchmarks/REDS-Val4"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:urllib3 v2 only supports OpenSSL,ignore:resource_tracker:UserWarning}"
"$PYTHON_BIN" mcft_mixed_replay.py \
  --init mcft_net_confidence_reds_best.pth \
  --output "${OUTPUT:-mcft_net_confidence_mixed_replay_ssim.pth}" \
  --reds-gt "$ROOT/train_000_001_gt.mkv" --reds-lr "$ROOT/train_000_001_lr.mkv" \
  --reds-candidate "$ROOT/train_000_001_mcft.mkv" --reds-anchors "$ROOT/train_000_001_anchors.json" \
  --val-gt "$ROOT/006/gt.mkv" --val-lr "$ROOT/006/lr_x2.mkv" \
  --val-candidate "$ROOT/006/mcft_v3.mkv" --val-anchors "$ROOT/006/anchors_10pct.json" \
  --vimeo-triplets "${VIMEO_TRIPLETS:-100}" --epochs "${EPOCHS:-1}" \
  --shuffle-buffer "${SHUFFLE_BUFFER:-16}" --log-every "${LOG_EVERY:-8}" \
  --reds-steps-per-vimeo 3 --lr 5e-5 --distill-weight 0.10 \
  --edge-weight 0.05 --ssim-weight "${SSIM_WEIGHT:-0.05}" \
  --gradient-weight "${GRADIENT_WEIGHT:-0.15}" --texture-weight "${TEXTURE_WEIGHT:-0.0}" \
  --temporal-weight 0.10 --l2sp-weight 0.01 --device auto
