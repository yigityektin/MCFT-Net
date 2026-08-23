"""Rank MCFT failure cases and export diagnostic crops."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def read_video(path: str, max_frames: int = 0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def y_channel(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[..., 0]


def psnr_y(gt_y: np.ndarray, candidate_y: np.ndarray) -> float:
    mse = float(np.mean((gt_y.astype(np.float32) - candidate_y.astype(np.float32)) ** 2))
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-12)))


def edge_mae(gt_y: np.ndarray, candidate_y: np.ndarray) -> float:
    gt_f = gt_y.astype(np.float32)
    candidate_f = candidate_y.astype(np.float32)
    gt_mag = np.hypot(
        cv2.Sobel(gt_f, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gt_f, cv2.CV_32F, 0, 1, ksize=3),
    )
    candidate_mag = np.hypot(
        cv2.Sobel(candidate_f, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(candidate_f, cv2.CV_32F, 0, 1, ksize=3),
    )
    return float(np.mean(np.abs(gt_mag - candidate_mag)))


def temporal_delta_mae(
    previous_gt_y: np.ndarray | None,
    gt_y: np.ndarray,
    previous_candidate_y: np.ndarray | None,
    candidate_y: np.ndarray,
) -> float:
    if previous_gt_y is None or previous_candidate_y is None:
        return float("nan")
    return float(
        np.mean(
            np.abs(
                (candidate_y.astype(np.float32) - previous_candidate_y.astype(np.float32))
                - (gt_y.astype(np.float32) - previous_gt_y.astype(np.float32))
            )
        )
    )


def nearest_anchor_distance(index: int, anchors: list[int]) -> int:
    return min(abs(index - anchor) for anchor in anchors)


def worst_error_crop(
    gt_y: np.ndarray,
    candidate_y: np.ndarray,
    crop: int,
) -> tuple[int, int, float]:
    error = np.abs(gt_y.astype(np.float32) - candidate_y.astype(np.float32))
    score = cv2.boxFilter(error, cv2.CV_32F, (crop, crop), normalize=True)
    _, value, _, point = cv2.minMaxLoc(score)
    x = int(np.clip(point[0] - crop // 2, 0, gt_y.shape[1] - crop))
    y = int(np.clip(point[1] - crop // 2, 0, gt_y.shape[0] - crop))
    return x, y, float(value)


def make_panel(gt: np.ndarray, base: np.ndarray, candidate: np.ndarray, x: int, y: int, crop: int) -> np.ndarray:
    views = [gt, base, candidate]
    crops = [frame[y : y + crop, x : x + crop] for frame in views]
    crop_panel = np.concatenate(crops, axis=1)
    scale = min(1.0, 360.0 / max(gt.shape[:2]))
    thumbs = [cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for frame in views]
    full_panel = np.concatenate(thumbs, axis=1)
    if crop_panel.shape[1] != full_panel.shape[1]:
        crop_panel = cv2.resize(crop_panel, (full_panel.shape[1], int(crop_panel.shape[0] * full_panel.shape[1] / crop_panel.shape[1])), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([full_panel, crop_panel], axis=0)


def write_top_panels(
    rows: list[dict[str, float | int]],
    gt_frames: list[np.ndarray],
    base_frames: list[np.ndarray],
    candidate_frames: list[np.ndarray],
    output_dir: Path,
    top_k: int,
    crop: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = {
        "worst_candidate_psnr": sorted(rows, key=lambda r: float(r["candidate_psnr_y"])),
        "worst_candidate_edge": sorted(rows, key=lambda r: float(r["candidate_edge_mae"]), reverse=True),
        "worst_candidate_temporal": sorted(
            [r for r in rows if np.isfinite(float(r["candidate_temporal_delta_mae"]))],
            key=lambda r: float(r["candidate_temporal_delta_mae"]),
            reverse=True,
        ),
        "least_psnr_gain": sorted(rows, key=lambda r: float(r["psnr_gain_vs_base"])),
    }
    for category, selected in categories.items():
        for rank, row in enumerate(selected[:top_k], start=1):
            index = int(row["frame"])
            x = int(row["worst_crop_x"])
            y = int(row["worst_crop_y"])
            panel = make_panel(gt_frames[index], base_frames[index], candidate_frames[index], x, y, crop)
            path = output_dir / f"{category}_rank{rank:02d}_frame{index:03d}.png"
            cv2.imwrite(str(path), panel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-border", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    gt_frames = read_video(args.ground_truth, args.max_frames)
    base_frames = read_video(args.base, args.max_frames)
    candidate_frames = read_video(args.candidate, args.max_frames)
    n = min(len(gt_frames), len(base_frames), len(candidate_frames))
    if n == 0:
        raise RuntimeError("No aligned frames found")
    anchors = sorted(map(int, json.loads(Path(args.anchors).read_text())["anchor_indices"]))
    anchor_set = set(anchors)
    border = args.crop_border

    rows: list[dict[str, float | int]] = []
    previous_gt_y = previous_base_y = previous_candidate_y = None
    for index in range(n):
        gt_y_full = y_channel(gt_frames[index])
        base_y_full = y_channel(base_frames[index])
        candidate_y_full = y_channel(candidate_frames[index])
        if border:
            sl = np.s_[border:-border, border:-border]
            gt_y = gt_y_full[sl]
            base_y = base_y_full[sl]
            candidate_y = candidate_y_full[sl]
        else:
            gt_y, base_y, candidate_y = gt_y_full, base_y_full, candidate_y_full

        base_psnr = psnr_y(gt_y, base_y)
        candidate_psnr = psnr_y(gt_y, candidate_y)
        base_ssim = float(structural_similarity(gt_y, base_y, data_range=255))
        candidate_ssim = float(structural_similarity(gt_y, candidate_y, data_range=255))
        base_edge = edge_mae(gt_y, base_y)
        candidate_edge = edge_mae(gt_y, candidate_y)
        base_temporal = temporal_delta_mae(previous_gt_y, gt_y, previous_base_y, base_y)
        candidate_temporal = temporal_delta_mae(previous_gt_y, gt_y, previous_candidate_y, candidate_y)
        crop = min(args.crop_size, gt_y.shape[0], gt_y.shape[1])
        x, y, crop_error = worst_error_crop(gt_y, candidate_y, crop)
        rows.append(
            {
                "frame": index,
                "is_anchor": int(index in anchor_set),
                "nearest_anchor_distance": nearest_anchor_distance(index, anchors),
                "base_psnr_y": base_psnr,
                "candidate_psnr_y": candidate_psnr,
                "psnr_gain_vs_base": candidate_psnr - base_psnr,
                "base_ssim_y": base_ssim,
                "candidate_ssim_y": candidate_ssim,
                "ssim_gain_vs_base": candidate_ssim - base_ssim,
                "base_edge_mae": base_edge,
                "candidate_edge_mae": candidate_edge,
                "edge_mae_gain_vs_base": base_edge - candidate_edge,
                "base_temporal_delta_mae": base_temporal,
                "candidate_temporal_delta_mae": candidate_temporal,
                "temporal_mae_gain_vs_base": base_temporal - candidate_temporal,
                "worst_crop_x": x + border,
                "worst_crop_y": y + border,
                "worst_crop_mean_abs_y_error": crop_error,
            }
        )
        previous_gt_y = gt_y
        previous_base_y = base_y
        previous_candidate_y = candidate_y

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_top_panels(rows, gt_frames, base_frames, candidate_frames, Path(args.output_dir), args.top_k, args.crop_size)

    nonanchors = [row for row in rows if not row["is_anchor"]]
    regressions = [row for row in nonanchors if float(row["psnr_gain_vs_base"]) < 0]
    print(f"frames {len(rows)} nonanchors {len(nonanchors)} anchors {len(rows) - len(nonanchors)}")
    print("nonanchor PSNR gain mean", float(np.mean([row["psnr_gain_vs_base"] for row in nonanchors])))
    print("nonanchor SSIM gain mean", float(np.mean([row["ssim_gain_vs_base"] for row in nonanchors])))
    print("nonanchor edge MAE gain mean", float(np.mean([row["edge_mae_gain_vs_base"] for row in nonanchors])))
    print("nonanchor temporal MAE gain mean", float(np.nanmean([row["temporal_mae_gain_vs_base"] for row in nonanchors])))
    print("nonanchor PSNR regressions", len(regressions))
    worst = min(nonanchors, key=lambda row: float(row["psnr_gain_vs_base"]))
    print(
        "least PSNR gain frame",
        int(worst["frame"]),
        "gain",
        float(worst["psnr_gain_vs_base"]),
        "nearest_anchor_distance",
        int(worst["nearest_anchor_distance"]),
    )
    print(f"[OK] wrote {out}")
    print(f"[OK] wrote panels to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
