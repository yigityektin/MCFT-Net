"""Fetch selected official REDS validation clips and build a controlled x2 benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from remotezip import RemoteZip


DEFAULT_URL = "https://huggingface.co/datasets/snah/REDS/resolve/main/val_sharp.zip"


def extract(url: str, root: Path, clips: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with RemoteZip(url) as archive:
        members = archive.namelist()
        for clip in clips:
            prefix = f"val/val_sharp/{clip}/"
            selected = sorted(n for n in members if n.startswith(prefix) and n.endswith(".png"))
            if len(selected) != 100:
                raise RuntimeError(f"Expected 100 frames for {clip}, found {len(selected)}")
            output = root / clip / "gt_frames"
            output.mkdir(parents=True, exist_ok=True)
            for index, member in enumerate(selected):
                target = output / Path(member).name
                if not target.exists():
                    target.write_bytes(archive.read(member))
                if (index + 1) % 25 == 0:
                    print(f"[{clip}] downloaded {index + 1}/100", flush=True)


def writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    result = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    if not result.isOpened():
        raise RuntimeError(f"Cannot create {path}")
    return result


def prepare(root: Path, clips: list[str], fps: float) -> None:
    for clip in clips:
        directory = root / clip
        frames = sorted((directory / "gt_frames").glob("*.png"))
        first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        if w % 2 or h % 2:
            raise ValueError(f"Odd dimensions for {clip}: {w}x{h}")
        gt_out = writer(directory / "gt.mkv", fps, (w, h))
        lr_out = writer(directory / "lr_x2.mkv", fps, (w // 2, h // 2))
        bic_out = writer(directory / "bicubic_x2.mkv", fps, (w, h))
        for frame_path in frames:
            gt = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            lr = cv2.resize(gt, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
            bicubic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
            gt_out.write(gt); lr_out.write(lr); bic_out.write(bicubic)
        gt_out.release(); lr_out.release(); bic_out.release()
        anchors = np.linspace(0, len(frames) - 1, 10).round().astype(int).tolist()
        manifest = {
            "method": "uniform_10_percent_hr_reference_budget",
            "dataset": "official_REDS_validation_sharp",
            "clip": clip,
            "scale": 2,
            "degradation": "opencv_bicubic_downsample",
            "total_frames": len(frames),
            "anchor_indices": anchors,
        }
        (directory / "anchors_10pct.json").write_text(json.dumps(manifest, indent=2))
        print(f"[{clip}] prepared {len(frames)} frames, anchors={anchors}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--root", type=Path, default=Path("external_benchmarks/REDS-Val4"))
    parser.add_argument("--clips", nargs="+", default=["000", "001", "006", "017"])
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()
    extract(args.url, args.root, args.clips)
    prepare(args.root, args.clips, args.fps)


if __name__ == "__main__":
    main()
