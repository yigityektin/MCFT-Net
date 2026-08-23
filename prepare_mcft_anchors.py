"""Select auditable temporal anchors from an original video and extract them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_at(cap: cv2.VideoCapture, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {index}")
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", required=True)
    p.add_argument("--hat-video", required=True)
    p.add_argument("--count", type=int, default=15)
    p.add_argument("--output-dir", default="mcft_anchors")
    args = p.parse_args()
    original = cv2.VideoCapture(args.original)
    hat = cv2.VideoCapture(args.hat_video)
    if not original.isOpened() or not hat.isOpened():
        raise RuntimeError("Could not open original or HAT video")
    n = int(original.get(cv2.CAP_PROP_FRAME_COUNT))
    if n != int(hat.get(cv2.CAP_PROP_FRAME_COUNT)):
        raise ValueError("Original and HAT frame counts must match")
    indices = np.linspace(0, n - 1, min(max(args.count, 2), n), dtype=int).tolist()
    root = Path(args.output_dir); lr_dir = root / "original"; sr_dir = root / "hat"
    lr_dir.mkdir(parents=True, exist_ok=True); sr_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for rank, index in enumerate(indices):
        lr, sr = read_at(original, index), read_at(hat, index)
        lr_name, sr_name = f"anchor_{index:04d}.png", f"anchor_{index:04d}.png"
        if not cv2.imwrite(str(lr_dir / lr_name), lr):
            raise RuntimeError(f"Could not write original anchor {index}")
        if not cv2.imwrite(str(sr_dir / sr_name), sr):
            raise RuntimeError(f"Could not write HAT anchor {index}")
        frames.append({"rank": rank, "video_frame_index": index,
                       "original": str(lr_dir / lr_name), "hat": str(sr_dir / sr_name)})
    manifest = {"method": "uniform_temporal_anchors_from_original",
                "original_video": args.original, "hat_video": args.hat_video,
                "total_frames": n, "anchor_indices": indices, "frames": frames}
    path = root / "mcft_anchor_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    original.release(); hat.release()
    print(f"[OK] extracted {len(indices)} anchors selected from {args.original} -> {path}")


if __name__ == "__main__":
    main()
