"""Estimate blur and sampling phase from aligned HR anchors and an LR video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def degrade(image: np.ndarray, size: tuple[int, int], sigma: float,
            phase_x: float, phase_y: float, resize_kernel: str = "area") -> np.ndarray:
    work = image.astype(np.float32)
    if sigma > 1e-6:
        work = cv2.GaussianBlur(work, (0, 0), sigma)
    matrix = np.float32([[1, 0, phase_x], [0, 1, phase_y]])
    work = cv2.warpAffine(work, matrix, (work.shape[1], work.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    interpolation = cv2.INTER_CUBIC if resize_kernel == "bicubic" else cv2.INTER_AREA
    return cv2.resize(work, size, interpolation=interpolation)


def read_at(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok: raise RuntimeError(f"Cannot read frame {index}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-resolution", required=True)
    parser.add_argument("--low-resolution", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output", default="degradation_model.json")
    parser.add_argument("--resize-kernel", choices=("area", "bicubic"), default="area")
    args = parser.parse_args()
    indices = json.loads(Path(args.anchors).read_text())["anchor_indices"]
    hr_cap = cv2.VideoCapture(args.high_resolution)
    lr_cap = cv2.VideoCapture(args.low_resolution)
    lr_w = int(lr_cap.get(cv2.CAP_PROP_FRAME_WIDTH)); lr_h = int(lr_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pairs = []
    for index in indices:
        hr = cv2.cvtColor(read_at(hr_cap, index), cv2.COLOR_BGR2YCrCb)[..., 0]
        lr = cv2.cvtColor(read_at(lr_cap, index), cv2.COLOR_BGR2YCrCb)[..., 0]
        pairs.append((hr, lr.astype(np.float32)))
    candidates = []
    for sigma in np.arange(0.0, 1.51, 0.25):
        for phase_y in (-0.5, -0.25, 0.0, 0.25, 0.5):
            for phase_x in (-0.5, -0.25, 0.0, 0.25, 0.5):
                errors = []
                for hr, lr in pairs:
                    pred = degrade(hr, (lr_w, lr_h), float(sigma), phase_x, phase_y,
                                   args.resize_kernel)
                    errors.append(float(np.mean((pred[2:-2,2:-2]-lr[2:-2,2:-2])**2)))
                candidates.append((float(np.mean(errors)),float(sigma),phase_x,phase_y))
    mse, sigma, phase_x, phase_y = min(candidates)
    residuals = []
    for hr, lr in pairs:
        pred = degrade(hr,(lr_w,lr_h),sigma,phase_x,phase_y,args.resize_kernel)
        residuals.append(lr-pred)
    residual = np.concatenate([x.ravel()[::32] for x in residuals])
    model = {"blur_sigma_hr":sigma,"phase_x_hr":phase_x,"phase_y_hr":phase_y,
             "lr_mse":mse,"codec_residual_std":float(np.std(residual)),
             "anchor_count":len(indices),"resize_kernel":args.resize_kernel,
             "operator":f"gaussian_phase_{args.resize_kernel}"}
    Path(args.output).write_text(json.dumps(model,indent=2))
    print(json.dumps(model,indent=2))


if __name__ == "__main__": main()
