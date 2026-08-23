#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-hat/bin/python}"
# The system Python emits two cosmetic shutdown/network compatibility warnings.
# Streaming and checkpoint integrity are covered by the smoke/unit tests.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:urllib3 v2 only supports OpenSSL,ignore:resource_tracker:UserWarning}"

# The checkpoint is a candidate until it passes the fixed REDS validation/test gates.
"$PYTHON_BIN" mcft_hf_stream.py train \
  --init mcft_net_confidence_reds_best.pth \
  --output mcft_net_confidence_vimeo_pretrained.pth \
  --train-triplets "${TRAIN_TRIPLETS:-2000}" \
  --val-triplets "${VAL_TRIPLETS:-200}" \
  --shuffle-buffer "${SHUFFLE_BUFFER:-64}" \
  --log-every "${LOG_EVERY:-10}" \
  --epochs "${EPOCHS:-4}" \
  --patch 96 --patches-per-triplet 2 --val-patches-per-triplet 1 \
  --batch-size 8 --width 32 --blocks 6 --lr 1e-4 --device auto
