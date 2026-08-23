"""Motion-compensated high-frequency residual transfer for video SR.

The low-resolution video is used for optical flow.  ``--anchor-video`` must be
frame-aligned with it and contain enhanced/SR versions of the same frames.  Only
the selected anchor indices are read from that video; every other output frame
is a bicubic upscale plus confidence-masked detail warped from adjacent anchors.
"""

from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def read_anchor_indices(path: str | None, total: int, stride: int) -> list[int]:
    if path is None:
        indices = list(range(0, total, max(1, stride)))
        if total and indices[-1] != total - 1:
            indices.append(total - 1)
        return indices

    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        raw: Iterable[int] = obj
    elif "anchor_indices" in obj:
        raw = obj["anchor_indices"]
    elif "sampled_indices" in obj:
        raw = obj["sampled_indices"]
    elif "frames" in obj:
        raw = [f.get("video_frame_index", f.get("frame_index")) for f in obj["frames"]]
    else:
        raise ValueError("Anchor JSON needs a list, anchor_indices, sampled_indices, or frames")
    indices = sorted({int(i) for i in raw if i is not None and 0 <= int(i) < total})
    if not indices:
        raise ValueError("No valid anchor indices")
    return indices


def flow_farneback(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Dense forward flow: a pixel in src moves to its location in dst."""
    a = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        a, b, None, pyr_scale=0.5, levels=4, winsize=21, iterations=4,
        poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def flow_dis(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Dense inverse-search flow with subpixel variational refinement."""
    a = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    dis.setUseMeanNormalization(True)
    dis.setVariationalRefinementIterations(5)
    return dis.calc(a, b, None)


def compute_flow(src: np.ndarray, dst: np.ndarray, method: str) -> np.ndarray:
    if method == "raft":
        return flow_raft(src, dst)
    return flow_dis(src, dst) if method == "dis" else flow_farneback(src, dst)


_RAFT_MODEL = None


def flow_raft(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Torchvision RAFT-Small forward flow, padded to multiples of eight."""
    global _RAFT_MODEL
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    if _RAFT_MODEL is None:
        os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / ".torch-cache"))
        _RAFT_MODEL = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).eval()
        if torch.backends.mps.is_available(): _RAFT_MODEL = _RAFT_MODEL.to("mps")
    device = next(_RAFT_MODEL.parameters()).device
    h, w = src.shape[:2]; ph = (8-h%8)%8; pw = (8-w%8)%8
    def tensor(frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        value=torch.from_numpy(rgb).permute(2,0,1).float()/127.5-1.0
        return torch.nn.functional.pad(value[None],(0,pw,0,ph),mode="replicate").to(device)
    with torch.inference_mode(): flow=_RAFT_MODEL(tensor(src),tensor(dst))[-1][0,:,:h,:w]
    return flow.permute(1,2,0).float().cpu().numpy()


def remap_with_displacement(image: np.ndarray, displacement: np.ndarray,
                            border_mode: int = cv2.BORDER_REFLECT101,
                            interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Backward-sample image at p + displacement[p]."""
    h, w = displacement.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    return cv2.remap(
        image, (xx + displacement[..., 0]).astype(np.float32),
        (yy + displacement[..., 1]).astype(np.float32), interpolation,
        borderMode=border_mode,
    )


def flow_confidence(target: np.ndarray, anchor: np.ndarray,
                    target_to_anchor: np.ndarray, consistency_sigma: float,
                    photometric_sigma: float, flow_method: str = "farneback") -> np.ndarray:
    """Forward/backward consistency and brightness agreement in [0, 1]."""
    anchor_to_target = compute_flow(anchor, target, flow_method)
    back_at_match = remap_with_displacement(anchor_to_target, target_to_anchor,
                                            cv2.BORDER_CONSTANT)
    fb_error = np.linalg.norm(target_to_anchor + back_at_match, axis=2)
    geometric = np.exp(-0.5 * (fb_error / max(consistency_sigma, 1e-6)) ** 2)

    tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ag = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    warped_ag = remap_with_displacement(ag, target_to_anchor, cv2.BORDER_CONSTANT)
    photo_error = np.abs(tg - warped_ag)
    photometric = np.exp(-0.5 * (photo_error / max(photometric_sigma, 1e-6)) ** 2)

    h, w = tg.shape
    yy, xx = np.mgrid[:h, :w]
    mx = xx + target_to_anchor[..., 0]
    my = yy + target_to_anchor[..., 1]
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    confidence = geometric * photometric * valid
    return cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), 0.8)


def super_resolution_residual(anchor_sr: np.ndarray, anchor_lr: np.ndarray,
                              detail_sigma: float) -> np.ndarray:
    """Detail added by SR relative to the same frame's bicubic baseline.

    Using highpass(SR) here would add edges already present in the target and
    double-sharpen them.  High-passing only the SR-minus-bicubic delta removes
    slow brightness/color drift while retaining model-added detail.
    """
    h, w = anchor_sr.shape[:2]
    baseline = cv2.resize(anchor_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    sr_y = cv2.cvtColor(anchor_sr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    base_y = cv2.cvtColor(baseline, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    delta = sr_y - base_y
    low_delta = cv2.GaussianBlur(delta, (0, 0), max(detail_sigma, 1e-3))
    return delta - low_delta


def anchor_high_frequency(anchor_sr: np.ndarray, detail_sigma: float) -> np.ndarray:
    """High-frequency luminance carried by a super-resolved anchor frame."""
    y = cv2.cvtColor(anchor_sr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    low = cv2.GaussianBlur(y, (0, 0), max(detail_sigma, 1e-3))
    return y - low


def frequency_bands(frame: np.ndarray, fine_sigma: float,
                    coarse_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return low, middle, and high luminance bands that exactly reconstruct Y."""
    y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    fine_low = cv2.GaussianBlur(y, (0, 0), max(fine_sigma, 1e-3))
    coarse_low = cv2.GaussianBlur(y, (0, 0), max(coarse_sigma, fine_sigma + 1e-3))
    high = y - fine_low
    middle = fine_low - coarse_low
    return coarse_low, middle, high


def edge_orientation_agreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3); ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3); by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    dot = ax * bx + ay * by
    denom = np.sqrt((ax * ax + ay * ay) * (bx * bx + by * by)) + 1e-6
    # Flat regions should not be rejected merely because orientation is undefined.
    flat = (ax * ax + ay * ay < 1.0) & (bx * bx + by * by < 1.0)
    return np.where(flat, 1.0, np.clip(dot / denom, 0.0, 1.0)).astype(np.float32)


def overlapping_patch_agreement(a: np.ndarray, b: np.ndarray, radius: int) -> np.ndarray:
    """Continuous overlapping-patch normalized correlation in [0, 1]."""
    k = 2 * max(1, radius) + 1
    a = a.astype(np.float32); b = b.astype(np.float32)
    ma = cv2.boxFilter(a, cv2.CV_32F, (k, k), normalize=True)
    mb = cv2.boxFilter(b, cv2.CV_32F, (k, k), normalize=True)
    va = cv2.boxFilter(a * a, cv2.CV_32F, (k, k), normalize=True) - ma * ma
    vb = cv2.boxFilter(b * b, cv2.CV_32F, (k, k), normalize=True) - mb * mb
    cov = cv2.boxFilter(a * b, cv2.CV_32F, (k, k), normalize=True) - ma * mb
    corr = cov / (np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0)) + 1e-6)
    flat = (va < 1e-3) & (vb < 1e-3)
    return np.where(flat, 1.0, np.clip(corr, 0.0, 1.0)).astype(np.float32)


def scene_compatible(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ha = cv2.calcHist([ga], [0], None, [32], [0, 256]); cv2.normalize(ha, ha)
    hb = cv2.calcHist([gb], [0], None, [32], [0, 256]); cv2.normalize(hb, hb)
    return cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA) <= threshold


def phase_preserving_spectral_transfer(
    target: np.ndarray,
    anchor: np.ndarray,
    confidence: np.ndarray,
    patch_size: int,
    stride: int,
    min_gain: float,
    max_gain: float,
    high_frequency_start: float,
    min_correlation: float,
) -> np.ndarray:
    """Transfer bounded local magnitude while retaining target Fourier phase."""
    h, w = target.shape
    p = max(16, min(patch_size, h, w))
    step = max(1, min(stride, p))
    ys = list(range(0, max(h - p, 0) + 1, step)); xs = list(range(0, max(w - p, 0) + 1, step))
    if not ys or ys[-1] != h - p: ys.append(h - p)
    if not xs or xs[-1] != w - p: xs.append(w - p)
    win1 = np.hanning(p).astype(np.float32)
    window = np.outer(win1, win1)
    window2 = window * window
    yy, xx = np.mgrid[:p, :p]
    radius = np.sqrt((yy - (p - 1) / 2) ** 2 + (xx - (p - 1) / 2) ** 2)
    high_mask = radius >= high_frequency_start * radius.max()
    accum = np.zeros_like(target, np.float32); weights = np.zeros_like(target, np.float32)
    eps = 1e-6
    for y in ys:
        for x in xs:
            t = target[y:y+p, x:x+p].astype(np.float32)
            a = anchor[y:y+p, x:x+p].astype(np.float32)
            c = float(np.mean(confidence[y:y+p, x:x+p]))
            tz, az = t - t.mean(), a - a.mean()
            corr = float(np.sum(tz * az) / (np.linalg.norm(tz) * np.linalg.norm(az) + eps))
            if corr <= min_correlation or c <= 1e-4:
                proposal = t
            else:
                c *= np.clip((corr - min_correlation) / max(1.0 - min_correlation, eps), 0, 1)
                ft = np.fft.fftshift(np.fft.fft2(t * window))
                fa = np.fft.fftshift(np.fft.fft2(a * window))
                mt, ma = np.abs(ft), np.abs(fa)
                ratio = np.clip(ma / (mt + eps), min_gain, max_gain)
                ratio = np.where(high_mask, 1.0 + c * (ratio - 1.0), 1.0)
                changed = ft * ratio  # target complex phase is retained exactly
                proposal = np.fft.ifft2(np.fft.ifftshift(changed)).real / np.maximum(window, 1e-3)
            accum[y:y+p, x:x+p] += proposal.astype(np.float32) * window2
            weights[y:y+p, x:x+p] += window2
    return np.where(weights > 1e-6, accum / np.maximum(weights, 1e-6), target).astype(np.float32)


def resize_flow(flow: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = flow.shape[:2]
    out = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    out[..., 0] *= width / w
    out[..., 1] *= height / h
    return out


def resize_for_flow(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (max(16, round(w * scale)), max(16, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def adjacent_anchors(frame_index: int, anchors: list[int]) -> list[int]:
    pos = bisect_left(anchors, frame_index)
    result = []
    if pos > 0:
        result.append(anchors[pos - 1])
    if pos < len(anchors):
        result.append(anchors[pos])
    return list(dict.fromkeys(result))


def read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {index}")
    return frame


def run(args: argparse.Namespace) -> None:
    low_cap = cv2.VideoCapture(args.input)
    anchor_cap = cv2.VideoCapture(args.anchor_video)
    if not low_cap.isOpened() or not anchor_cap.isOpened():
        raise RuntimeError("Could not open input or anchor video")

    total = int(low_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anchor_total = int(anchor_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0 or anchor_total != total:
        raise ValueError(f"Videos must have the same known frame count ({total} != {anchor_total})")
    fps = low_cap.get(cv2.CAP_PROP_FPS) or 30.0
    anchor_fps = anchor_cap.get(cv2.CAP_PROP_FPS) or fps
    if abs(anchor_fps - fps) > 1e-3:
        raise ValueError(f"Videos must have the same FPS ({fps} != {anchor_fps})")
    anchors = read_anchor_indices(args.anchors, total, args.anchor_stride)

    first_lr = read_frame(low_cap, 0)
    first_sr = read_frame(anchor_cap, anchors[0])
    out_h, out_w = first_sr.shape[:2]
    if first_lr.shape[0] > out_h or first_lr.shape[1] > out_w:
        raise ValueError("Anchor video must not be smaller than the low-resolution video")
    lr_aspect = first_lr.shape[1] / first_lr.shape[0]
    sr_aspect = out_w / out_h
    if abs(lr_aspect - sr_aspect) > 1e-3:
        raise ValueError("Input and anchor video must have the same aspect ratio")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {out_path}")

    lr_cache: dict[int, np.ndarray] = {}
    sr_cache: dict[int, np.ndarray] = {}

    def lr_at(i: int) -> np.ndarray:
        if i not in lr_cache:
            lr_cache[i] = read_frame(low_cap, i)
        return lr_cache[i]

    def sr_at(i: int) -> np.ndarray:
        if i not in sr_cache:
            frame = read_frame(anchor_cap, i)
            if frame.shape[:2] != (out_h, out_w):
                raise ValueError(f"Anchor frame {i} has an inconsistent size")
            sr_cache[i] = frame
        return sr_cache[i]

    anchor_set = set(anchors)
    for i in range(total):
        target_lr = lr_at(i)
        baseline = cv2.resize(target_lr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if i in anchor_set:
            output = sr_at(i)
        else:
            detail_sum = np.zeros((out_h, out_w), np.float32)
            weight_sum = np.zeros((out_h, out_w), np.float32)
            if args.residual_mode == "band-replace":
                _, target_middle, target_high = frequency_bands(
                    baseline, args.fine_sigma, args.coarse_sigma
                )
                middle_sum = np.zeros_like(detail_sum)
                high_sum = np.zeros_like(detail_sum)
                middle_weights = np.zeros_like(detail_sum)
                high_weights = np.zeros_like(detail_sum)
            elif args.residual_mode == "phase-preserve":
                spectral_anchor_sum = np.zeros_like(detail_sum)
                spectral_weights = np.zeros_like(detail_sum)
            target_flow_frame = resize_for_flow(target_lr, args.flow_scale)
            for a in adjacent_anchors(i, anchors):
                distance = abs(i - a)
                if distance > args.max_distance:
                    continue
                anchor_lr = lr_at(a)
                if not scene_compatible(target_lr, anchor_lr, args.scene_threshold):
                    continue
                anchor_flow_frame = resize_for_flow(anchor_lr, args.flow_scale)
                t_to_a = compute_flow(target_flow_frame, anchor_flow_frame, args.flow_method)
                confidence = flow_confidence(
                    target_flow_frame, anchor_flow_frame, t_to_a,
                    args.consistency_sigma, args.photometric_sigma, args.flow_method,
                )
                temporal = np.exp(-distance / max(args.temporal_decay, 1e-6))
                confidence *= temporal
                flow_sr = resize_flow(t_to_a, out_w, out_h)
                conf_sr = cv2.resize(confidence, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                if args.residual_mode == "phase-preserve":
                    anchor_y = cv2.cvtColor(sr_at(a), cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
                    warped_y = remap_with_displacement(
                        anchor_y, flow_sr, cv2.BORDER_CONSTANT, cv2.INTER_CUBIC
                    )
                    motion_mag = cv2.resize(np.linalg.norm(t_to_a, axis=2),
                                            (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    ws = conf_sr * np.exp(-args.spectral_motion_decay * motion_mag)
                    spectral_anchor_sum += ws * warped_y
                    spectral_weights += ws
                    continue
                if args.residual_mode == "band-replace":
                    _, anchor_middle, anchor_high = frequency_bands(
                        sr_at(a), args.fine_sigma, args.coarse_sigma
                    )
                    warped_middle = remap_with_displacement(
                        anchor_middle, flow_sr, cv2.BORDER_CONSTANT,
                        cv2.INTER_CUBIC if args.frequency_warp == "cubic" else cv2.INTER_LINEAR,
                    )
                    warped_high = remap_with_displacement(
                        anchor_high, flow_sr, cv2.BORDER_CONSTANT,
                        cv2.INTER_CUBIC if args.frequency_warp == "cubic" else cv2.INTER_LINEAR,
                    )
                    # Disagreement means the warped frequency is probably from
                    # a different edge, an occlusion, or inaccurate motion.
                    middle_agree = np.exp(
                        -np.abs(warped_middle - target_middle)
                        / max(args.middle_agreement, 1e-6)
                    )
                    high_agree = np.exp(
                        -np.abs(warped_high - target_high)
                        / max(args.high_agreement, 1e-6)
                    )
                    edge_middle = edge_orientation_agreement(target_middle, warped_middle)
                    edge_high = edge_orientation_agreement(target_high, warped_high)
                    patch_middle = overlapping_patch_agreement(
                        target_middle, warped_middle, args.patch_radius
                    )
                    patch_high = overlapping_patch_agreement(
                        target_high, warped_high, args.patch_radius
                    )
                    motion_mag = cv2.resize(np.linalg.norm(t_to_a, axis=2),
                                            (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    wm = conf_sr * middle_agree * edge_middle * patch_middle * np.exp(-args.middle_motion_decay * motion_mag)
                    wh = conf_sr * high_agree * edge_high * patch_high * np.exp(-args.high_motion_decay * motion_mag)
                    middle_sum += wm * warped_middle
                    high_sum += wh * warped_high
                    middle_weights += wm
                    high_weights += wh
                    continue
                elif args.residual_mode == "highpass":
                    anchor_detail = anchor_high_frequency(sr_at(a), args.detail_sigma)
                else:
                    anchor_detail = super_resolution_residual(
                        sr_at(a), anchor_lr, args.detail_sigma
                    )
                warped_detail = remap_with_displacement(
                    anchor_detail, flow_sr, cv2.BORDER_CONSTANT,
                )
                detail_sum += conf_sr * warped_detail
                weight_sum += conf_sr

            ycc = cv2.cvtColor(baseline, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            if args.residual_mode == "phase-preserve":
                fused_anchor = spectral_anchor_sum / np.maximum(spectral_weights, 1e-6)
                target_y = ycc[..., 0].copy()
                ycc[..., 0] = np.clip(phase_preserving_spectral_transfer(
                    target_y, fused_anchor, np.clip(spectral_weights, 0, 1),
                    args.spectral_patch, args.spectral_stride,
                    args.spectral_min_gain, args.spectral_max_gain,
                    args.spectral_hf_start, args.spectral_min_correlation,
                ), 0, 255)
            elif args.residual_mode == "band-replace":
                predicted_middle = middle_sum / np.maximum(middle_weights, 1e-6)
                predicted_high = high_sum / np.maximum(high_weights, 1e-6)
                cm = np.clip(middle_weights, 0.0, 1.0)
                ch = np.clip(high_weights, 0.0, 1.0)
                correction = (
                    args.middle_weight * cm * (predicted_middle - target_middle)
                    + args.high_weight * ch * (predicted_high - target_high)
                )
                ycc[..., 0] = np.clip(ycc[..., 0] + correction, 0, 255)
            else:
                fused_detail = detail_sum / np.maximum(weight_sum, 1e-6)
                strength = args.strength * np.clip(weight_sum, 0.0, 1.0)
                ycc[..., 0] = np.clip(ycc[..., 0] + strength * fused_detail, 0, 255)
            output = cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        writer.write(output)

        keep = set(adjacent_anchors(i + 1, anchors)) | {i, i + 1}
        lr_cache = {k: v for k, v in lr_cache.items() if k in keep}
        sr_cache = {k: v for k, v in sr_cache.items() if k in keep}
        if (i + 1) % 20 == 0 or i + 1 == total:
            print(f"[MCFT] {i + 1}/{total}")

    writer.release()
    low_cap.release()
    anchor_cap.release()
    print(f"[OK] wrote {out_path} using {len(anchors)} anchors")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Low-resolution input video")
    p.add_argument("--anchor-video", required=True, help="Frame-aligned enhanced/SR video")
    p.add_argument("--output", required=True)
    p.add_argument("--anchors", help="Anchor index JSON; defaults to regular spacing")
    p.add_argument("--anchor-stride", type=int, default=10)
    p.add_argument("--max-distance", type=int, default=15)
    p.add_argument("--temporal-decay", type=float, default=8.0)
    p.add_argument("--detail-sigma", type=float, default=1.2)
    p.add_argument("--strength", type=float, default=0.8)
    p.add_argument("--consistency-sigma", type=float, default=1.5)
    p.add_argument("--photometric-sigma", type=float, default=0.08)
    p.add_argument("--flow-scale", type=float, default=0.5,
                   help="Flow resolution relative to LR input (default: 0.5)")
    p.add_argument("--flow-method", choices=("farneback", "dis"), default="farneback")
    p.add_argument("--frequency-warp", choices=("linear", "cubic"), default="linear")
    p.add_argument("--residual-mode", choices=("phase-preserve", "band-replace", "highpass", "sr-delta"),
                   default="band-replace",
                   help="band-replace is improved MCFT; other modes are ablations")
    p.add_argument("--fine-sigma", type=float, default=0.8)
    p.add_argument("--coarse-sigma", type=float, default=2.4)
    p.add_argument("--middle-weight", type=float, default=0.65)
    p.add_argument("--high-weight", type=float, default=0.20)
    p.add_argument("--middle-agreement", type=float, default=10.0)
    p.add_argument("--high-agreement", type=float, default=6.0)
    p.add_argument("--middle-motion-decay", type=float, default=0.08)
    p.add_argument("--high-motion-decay", type=float, default=0.20)
    p.add_argument("--scene-threshold", type=float, default=0.45)
    p.add_argument("--patch-radius", type=int, default=5,
                   help="Radius of continuous overlapping-patch agreement")
    p.add_argument("--spectral-patch", type=int, default=64)
    p.add_argument("--spectral-stride", type=int, default=32)
    p.add_argument("--spectral-min-gain", type=float, default=0.90)
    p.add_argument("--spectral-max-gain", type=float, default=1.12)
    p.add_argument("--spectral-hf-start", type=float, default=0.25)
    p.add_argument("--spectral-min-correlation", type=float, default=0.65)
    p.add_argument("--spectral-motion-decay", type=float, default=0.20)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
