from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="RealESRGAN_x2plus.pth")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # BasicSR 1.4.x imports a removed torchvision compatibility module.
    import torchvision.transforms.functional as functional
    import sys
    sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2, model_path=args.model, model=model,
        tile=args.tile, tile_pad=10, pre_pad=0,
        half=(device.type == "cuda"), device=device,
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="realesrgan_frames_") as tmp:
        tmp_path = Path(tmp)
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sr, _ = upsampler.enhance(frame, outscale=2)
            cv2.imwrite(str(tmp_path / f"{index + 1:08d}.png"), sr)
            index += 1
            print(f"{index}/{total}", flush=True)
        cap.release()
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps),
            "-i", str(tmp_path / "%08d.png"),
            "-c:v", "ffv1", "-pix_fmt", "yuv444p", str(output),
        ], check=True)
    print(f"wrote {output} {index} frames")


if __name__ == "__main__":
    main()
