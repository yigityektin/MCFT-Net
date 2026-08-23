"""Select temporally distributed, sharp, low-motion anchors from an original video."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np


def norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [5, 95])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--output", default="adaptive_anchor_manifest.json")
    p.add_argument("--scene-threshold", type=float, default=0.45)
    args = p.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {args.video}")
    sharp, motion, cuts, previous, index = [], [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        if previous is None:
            motion.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            motion.append(float(np.median(np.linalg.norm(flow, axis=2))))
            h0 = cv2.calcHist([previous], [0], None, [32], [0, 256]); cv2.normalize(h0, h0)
            h1 = cv2.calcHist([gray], [0], None, [32], [0, 256]); cv2.normalize(h1, h1)
            if cv2.compareHist(h0, h1, cv2.HISTCMP_BHATTACHARYYA) > args.scene_threshold:
                cuts.append(index)
        previous, index = gray, index + 1
    cap.release(); n = len(sharp)
    if not n: raise RuntimeError("No frames")
    quality = norm(np.asarray(sharp)) - 0.35 * norm(np.asarray(motion))
    selected = {0, n - 1, *cuts}
    # One best candidate per temporal bin guarantees coverage while adapting locally.
    for ids in np.array_split(np.arange(n), max(1, args.count)):
        if ids.size: selected.add(int(ids[np.argmax(quality[ids])]))
    if len(selected) > args.count:
        mandatory = {0, n - 1, *cuts}
        optional = sorted(selected - mandatory, key=lambda i: quality[i], reverse=True)
        selected = mandatory | set(optional[:max(0, args.count - len(mandatory))])
    indices = sorted(selected)
    data = {"method": "adaptive_sharp_low_motion_temporal_coverage",
            "original_video": args.video, "total_frames": n,
            "scene_cuts": cuts, "anchor_indices": indices,
            "frames": [{"video_frame_index": i, "sharpness": sharp[i],
                        "motion": motion[i], "quality": float(quality[i])} for i in indices]}
    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] selected {len(indices)} anchors, cuts={cuts}, indices={indices}")


if __name__ == "__main__": main()
