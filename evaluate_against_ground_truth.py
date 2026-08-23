"""Evaluate one or more aligned videos against actual ground truth."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--candidate", action="append", required=True, help="NAME=VIDEO (repeatable)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--crop-border", type=int, default=2)
    p.add_argument("--output", default="ground_truth_quality.csv")
    p.add_argument("--anchors", help="Optional anchor manifest for non-anchor summary")
    args = p.parse_args()
    candidates = dict(x.split("=", 1) for x in args.candidate)
    anchors = set()
    if args.anchors:
        anchors = set(map(int, json.loads(Path(args.anchors).read_text(encoding="utf-8"))["anchor_indices"]))
    gt_cap = cv2.VideoCapture(args.ground_truth)
    caps = {k: cv2.VideoCapture(v) for k, v in candidates.items()}
    if not gt_cap.isOpened() or not all(c.isOpened() for c in caps.values()):
        raise RuntimeError("Could not open all videos")
    rows, i = [], 0
    previous_gt = None
    previous_candidates = {}
    while True:
        ok, gt = gt_cap.read()
        reads = {k: c.read() for k, c in caps.items()}
        if not ok or not all(x[0] for x in reads.values()): break
        if i % args.stride: i += 1; continue
        gy = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[..., 0]
        b = args.crop_border
        if b: gy = gy[b:-b, b:-b]
        for name, (_, frame) in reads.items():
            if frame.shape[:2] != gt.shape[:2]:
                raise ValueError(f"{name} size differs at frame {i}")
            y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[..., 0]
            if b: y = y[b:-b, b:-b]
            mse = float(np.mean((gy.astype(np.float32) - y.astype(np.float32)) ** 2))
            gyf = gy.astype(np.float32); yf = y.astype(np.float32)
            gx = cv2.Sobel(gyf, cv2.CV_32F, 1, 0, ksize=3)
            gy_edge = cv2.Sobel(gyf, cv2.CV_32F, 0, 1, ksize=3)
            yx = cv2.Sobel(yf, cv2.CV_32F, 1, 0, ksize=3)
            yy = cv2.Sobel(yf, cv2.CV_32F, 0, 1, ksize=3)
            edge_mae = float(np.mean(np.abs(np.hypot(gx, gy_edge) - np.hypot(yx, yy))))
            temporal = float("nan")
            if previous_gt is not None and name in previous_candidates:
                temporal = float(np.mean(np.abs(
                    (yf - previous_candidates[name]) - (gyf - previous_gt)
                )))
            rows.append({"frame": i, "is_anchor": int(i in anchors), "method": name,
                         "psnr_y": 10 * np.log10(255 ** 2 / max(mse, 1e-12)),
                         "ssim_y": structural_similarity(gy, y, data_range=255),
                         "laplacian_energy": float(cv2.Laplacian(y, cv2.CV_32F).var()),
                         "edge_mae": edge_mae,
                         "temporal_delta_mae": temporal})
            previous_candidates[name] = yf
        previous_gt = gy.astype(np.float32)
        i += 1
    gt_cap.release(); [c.release() for c in caps.values()]
    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    for name in candidates:
        subset = [r for r in rows if r["method"] == name]
        print(name, "frames", len(subset),
              "PSNR-Y", np.mean([r["psnr_y"] for r in subset]),
              "SSIM-Y", np.mean([r["ssim_y"] for r in subset]),
              "Laplacian", np.mean([r["laplacian_energy"] for r in subset]),
              "Edge-MAE", np.mean([r["edge_mae"] for r in subset]),
              "Temporal-delta-MAE", np.nanmean([r["temporal_delta_mae"] for r in subset]))
        if anchors:
            non = [r for r in subset if not r["is_anchor"]]
            print(name, "NONANCHOR frames", len(non),
                  "PSNR-Y", np.mean([r["psnr_y"] for r in non]),
                  "SSIM-Y", np.mean([r["ssim_y"] for r in non]),
                  "Laplacian", np.mean([r["laplacian_energy"] for r in non]),
                  "Edge-MAE", np.mean([r["edge_mae"] for r in non]),
                  "Temporal-delta-MAE", np.nanmean([r["temporal_delta_mae"] for r in non]))
    print(f"[OK] wrote {out}")


if __name__ == "__main__": main()
