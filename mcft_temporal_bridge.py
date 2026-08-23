"""Reduce MCFT sharpness pulses by bridging reconstructed frames into anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mcft_constrained_reconstruct import project_observation
from motion_frequency_transfer import (
    compute_flow,
    flow_confidence,
    remap_with_displacement,
    resize_flow,
)


def read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index}")
    return frame


def bridge_luma(
    frame: np.ndarray,
    observed_lr: np.ndarray,
    anchor: np.ndarray,
    strength: float,
    projection_strength: float,
) -> np.ndarray:
    """Blend a motion-aligned anchor under confidence and LR data constraints."""
    height, width = frame.shape[:2]
    target_gray = cv2.cvtColor(observed_lr, cv2.COLOR_BGR2GRAY)
    anchor_lr = cv2.resize(anchor, (observed_lr.shape[1], observed_lr.shape[0]),
                           interpolation=cv2.INTER_AREA)
    flow = compute_flow(cv2.cvtColor(target_gray, cv2.COLOR_GRAY2BGR), anchor_lr, "dis")
    confidence = flow_confidence(
        cv2.cvtColor(target_gray, cv2.COLOR_GRAY2BGR), anchor_lr, flow, 1.5, 0.08, "dis"
    )
    flow_hr = resize_flow(flow, width, height)
    confidence_hr = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    anchor_y = cv2.cvtColor(anchor, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    warped_y = remap_with_displacement(
        anchor_y, flow_hr, cv2.BORDER_REFLECT101, cv2.INTER_CUBIC
    )
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    current_y = ycc[..., 0].astype(np.float32)
    weight = np.clip(strength * confidence_hr, 0.0, strength)
    bridged = (1.0 - weight) * current_y + weight * warped_y
    bridged = project_observation(bridged, target_gray.astype(np.float32), projection_strength)
    ycc[..., 0] = np.clip(bridged, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def adaptive_anchor_detail(previous: np.ndarray | None, anchor: np.ndarray,
                           smooth: np.ndarray, following: np.ndarray | None = None) -> float:
    """Match anchor sharpness to its predecessor within a conservative range."""
    if previous is None:
        return 0.90
    references=[previous] if following is None else [previous,following]
    target=float(np.mean([np.sqrt(max(cv2.Laplacian(
        cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY).astype(np.float32),cv2.CV_32F
    ).var(),0.0)) for reference in references]))
    anchor_y = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth_y = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high = np.sqrt(max(cv2.Laplacian(anchor_y, cv2.CV_32F).var(), 0.0))
    low = np.sqrt(max(cv2.Laplacian(smooth_y, cv2.CV_32F).var(), 0.0))
    return float(np.clip((target-low)/max(high-low,1e-6),0.85,0.98))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--low-resolution", required=True)
    parser.add_argument("--ground-truth", required=True, help="Source of original anchor frames")
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.30)
    parser.add_argument("--projection-strength", type=float, default=0.8)
    parser.add_argument("--anchor-detail", default="1.0",
                        help="Original-anchor detail, 'auto', or 'auto-bidirectional'")
    args = parser.parse_args()

    anchor_indices = sorted(map(int, json.loads(Path(args.anchors).read_text())["anchor_indices"]))
    anchor_set = set(anchor_indices)
    source = cv2.VideoCapture(args.input)
    low_res = cv2.VideoCapture(args.low_resolution)
    ground_truth = cv2.VideoCapture(args.ground_truth)
    lookahead = cv2.VideoCapture(args.input)
    count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = source.get(cv2.CAP_PROP_FPS)
    codec = "FFV1" if Path(args.output).suffix.lower() in {".mkv", ".avi"} else "mp4v"
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open {args.output}")
    anchor_cache = {index: read_frame(ground_truth, index) for index in anchor_indices}

    previous_result = None
    for index in range(count):
        ok, frame = source.read()
        ok_lr, observed = low_res.read()
        if not ok or not ok_lr:
            raise RuntimeError(f"Sequential decode failed at frame {index}")
        if index in anchor_set:
            anchor = anchor_cache[index]
            if args.anchor_detail in {"auto","auto-bidirectional"}:
                smooth = cv2.resize(observed, (width, height), interpolation=cv2.INTER_CUBIC)
                following=None
                if args.anchor_detail=="auto-bidirectional" and index+1<count:
                    following=read_frame(lookahead,index+1)
                detail = adaptive_anchor_detail(previous_result, anchor, smooth, following)
                result = cv2.addWeighted(anchor, detail, smooth, 1.0-detail, 0.0)
            elif float(args.anchor_detail) < 1.0:
                detail = float(args.anchor_detail)
                smooth = cv2.resize(observed, (width, height), interpolation=cv2.INTER_CUBIC)
                result = cv2.addWeighted(anchor, detail, smooth, 1.0-detail, 0.0)
            else:
                result = anchor
        else:
            upcoming = next((a for a in anchor_indices if a > index), None)
            distance = upcoming - index if upcoming is not None else args.radius + 1
            if distance <= args.radius:
                anchor = anchor_cache[upcoming]
                taper = (args.radius + 1 - distance) / args.radius
                result = bridge_luma(
                    frame, observed, anchor, args.strength * taper, args.projection_strength
                )
            else:
                result = frame
        writer.write(result)
        previous_result = result
        print(f"[{index + 1}/{count}] frame {index}")

    source.release()
    low_res.release()
    ground_truth.release()
    lookahead.release()
    writer.release()


if __name__ == "__main__":
    main()
