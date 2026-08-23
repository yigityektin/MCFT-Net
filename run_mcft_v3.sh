#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
"$PYTHON_BIN" estimate_degradation.py --high-resolution BigBuckBunny_5s.mp4 --low-resolution BigBuckBunny_5s_LR540.mp4 --anchors controlled_adaptive_anchors_30.json --output estimated_degradation.json
"$PYTHON_BIN" mcft_constrained_reconstruct.py --ground-truth BigBuckBunny_5s.mp4 --low-resolution BigBuckBunny_5s_LR540.mp4 --anchors controlled_adaptive_anchors_30.json --output-dir mcft_v3_lossless_metrics --output-video BBB_5s_MCFT_v3_operator_lossless.mkv --gradient-source residual-multiscale --chroma-strength 0.125 --degradation-model estimated_degradation.json --flow-method dis --local-refinement phase
"$PYTHON_BIN" mcft_temporal_bridge.py --input BBB_5s_MCFT_v3_operator_lossless.mkv --low-resolution BigBuckBunny_5s_LR540.mp4 --ground-truth BigBuckBunny_5s.mp4 --anchors controlled_adaptive_anchors_30.json --output BBB_5s_MCFT_v3_anchor_auto_lossless.mkv --radius 0 --strength 0 --anchor-detail auto
ffmpeg -y -v error -i BBB_5s_MCFT_v3_anchor_auto_lossless.mkv -an -c:v libx264 -preset medium -b:v 28M -maxrate 28M -bufsize 56M -pix_fmt yuv420p BBB_5s_MCFT_v3_anchor_auto_28M.mp4
"$PYTHON_BIN" evaluate_against_ground_truth.py --ground-truth BigBuckBunny_5s.mp4 --candidate bicubic=BBB_5s_Bicubic_ratematched_28M.mp4 --candidate v3=BBB_5s_MCFT_v3_anchor_auto_28M.mp4 --anchors controlled_adaptive_anchors_30.json --output mcft_v3_final_quality.csv
