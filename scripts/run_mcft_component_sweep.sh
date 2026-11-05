#!/usr/bin/env bash
set -euo pipefail

# E6/E7 deterministic MCFT sweep. Run this on a tuning clip only; select the
# configuration using its validation metrics, then evaluate once on test clips.
: "${GT:?set GT}"
: "${LR:?set LR}"
: "${BASE:?set BASE}"
: "${ANCHORS:?set ANCHORS}"
: "${OUT_DIR:?set OUT_DIR}"

mkdir -p "$OUT_DIR"
common=(--ground-truth "$GT" --low-resolution "$LR" --base-video "$BASE"
  --anchor-source-video "$GT" --anchors "$ANCHORS" --max-frames "${MAX_FRAMES:-100}"
  --no-diagnostics --multiframe-radius 2 --multiframe-strength 0.2)

run_case() {
  local tag="$1"; shift
  python mcft_constrained_reconstruct.py "${common[@]}" "$@" \
    --output-video "$OUT_DIR/${tag}.mkv" --output-dir "$OUT_DIR/${tag}_diag"
}

# E6: one-factor component ablations.
run_case full
run_case no_local --local-refinement none
run_case multiscale --gradient-source residual-multiscale
run_case no_projection --projection-strength 0
run_case no_multiframe --multiframe-radius 0 --multiframe-strength 0

# E7: compact sensitivity sweep around the default projection/Poisson settings.
for projection in 0.4 0.8 1.2; do
  for lambda in 0.175 0.35 0.525; do
    run_case "p${projection}_l${lambda}" \
      --projection-strength "$projection" --lambda-gradient "$lambda"
  done
done
