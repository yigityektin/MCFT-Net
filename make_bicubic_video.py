"""Create the codec-matched bicubic baseline used by the MCFT pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    args = p.parse_args()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {out}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_CUBIC))
        count += 1
    cap.release(); writer.release()
    if total > 0 and count != total:
        raise RuntimeError(f"Decoded {count}/{total} frames")
    print(f"[OK] wrote {out}: {count} frames at {args.width}x{args.height}")


if __name__ == "__main__":
    main()
