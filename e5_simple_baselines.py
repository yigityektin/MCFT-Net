"""E5 controls: nearest-anchor warp and warp plus LR observation projection."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import cv2
import numpy as np

from motion_frequency_transfer import compute_flow, remap_with_displacement, resize_flow
from mcft_constrained_reconstruct import project_observation_operator


def read_at(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index}")
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--low-resolution", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--output-warp", required=True)
    p.add_argument("--output-project", required=True)
    p.add_argument("--projection-strength", type=float, default=1.0)
    args = p.parse_args()

    anchors = sorted(json.loads(Path(args.anchors).read_text())["anchor_indices"])
    if not anchors:
        raise ValueError("E5 requires at least one anchor")
    aset = set(anchors)
    gt_cap = cv2.VideoCapture(args.ground_truth)
    lr_cap = cv2.VideoCapture(args.low_resolution)
    if not gt_cap.isOpened() or not lr_cap.isOpened():
        raise RuntimeError("Could not open input videos")
    n = int(gt_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = gt_cap.get(cv2.CAP_PROP_FPS)
    h = int(gt_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(gt_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    warp_writer = cv2.VideoWriter(args.output_warp, fourcc, fps, (w, h))
    project_writer = cv2.VideoWriter(args.output_project, fourcc, fps, (w, h))
    anchor_cache = {a: read_at(gt_cap, a) for a in anchors}

    for i in range(n):
        gt = read_at(gt_cap, i)
        lr = read_at(lr_cap, i)
        if i in aset:
            warp_writer.write(anchor_cache[i])
            project_writer.write(anchor_cache[i])
            continue
        nearest = min(anchors, key=lambda a: abs(a - i))
        anchor = anchor_cache[nearest]
        target_gray = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY)
        anchor_lr = cv2.resize(anchor, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_AREA)
        flow = compute_flow(
            cv2.cvtColor(target_gray, cv2.COLOR_GRAY2BGR),
            anchor_lr,
            "dis",
        )
        flow_hr = resize_flow(flow, w, h)
        warped = remap_with_displacement(
            anchor.astype(np.float32), flow_hr,
            cv2.BORDER_REFLECT101, cv2.INTER_CUBIC,
        )
        warped = np.clip(warped, 0, 255).astype(np.uint8)
        wy = cv2.cvtColor(warped, cv2.COLOR_BGR2YCrCb)
        y = wy[..., 0].astype(np.float32)
        projected_y = project_observation_operator(
            y, target_gray.astype(np.float32), args.projection_strength, None
        )
        projected = wy.copy()
        projected[..., 0] = np.clip(projected_y, 0, 255).astype(np.uint8)
        projected = cv2.cvtColor(projected, cv2.COLOR_YCrCb2BGR)
        warp_writer.write(warped)
        project_writer.write(projected)
        print(f"[{i + 1}/{n}]")

    gt_cap.release(); lr_cap.release()
    warp_writer.release(); project_writer.release()


if __name__ == "__main__":
    main()
