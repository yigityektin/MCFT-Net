"""Stream Vimeo-90K-mini and pretrain the protected MCFT confidence refiner.

The remote dataset is never materialized in full. For each Vimeo triplet, the
first and third sharp frames are local reference anchors and the middle frame
is the supervised target. This is pretraining, not a 10%-anchor benchmark.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from mcft_net import build_model, device_for, gradient_loss, make_confidence_input
from motion_frequency_transfer import compute_flow, flow_confidence, remap_with_displacement


DATASET_ID = "bijinc/vimeo-90k-mini"
ANCHORS = (0, 2)


def _bgr(image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def degrade_x2(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return bicubic x2 LR observation and its bicubic HR baseline."""
    h, w = frame.shape[:2]
    h -= h % 2
    w -= w % 2
    frame = frame[:h, :w]
    lr = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
    base = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    return lr, base


def _anchor_detail(anchor: np.ndarray, anchor_base: np.ndarray, sigma: float) -> np.ndarray:
    delta = anchor.astype(np.float32) - anchor_base.astype(np.float32)
    return delta - cv2.GaussianBlur(delta, (0, 0), sigma)


def make_triplet_example(images, flow_method="dis", strength=0.5, detail_sigma=0.8):
    """Build (network feature, GT, LR, proposal) for one Vimeo triplet."""
    frames = [_bgr(image) if not isinstance(image, np.ndarray) else image for image in images]
    h = min(frame.shape[0] for frame in frames)
    w = min(frame.shape[1] for frame in frames)
    h -= h % 2
    w -= w % 2
    frames = [np.ascontiguousarray(frame[:h, :w, :3], dtype=np.uint8) for frame in frames]
    degraded = [degrade_x2(frame) for frame in frames]
    lrs = [item[0] for item in degraded]
    bases = [item[1] for item in degraded]
    target_lr, target_base = lrs[1], bases[1]

    accumulated = np.zeros_like(target_base, dtype=np.float32)
    total_weight = np.zeros((h, w, 1), dtype=np.float32)
    for anchor_index in ANCHORS:
        flow = compute_flow(target_lr, lrs[anchor_index], flow_method)
        confidence = flow_confidence(
            target_lr, lrs[anchor_index], flow, 1.5, 0.08, flow_method
        )
        flow_hr = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
        detail = _anchor_detail(frames[anchor_index], bases[anchor_index], detail_sigma)
        warped = remap_with_displacement(detail, flow_hr, cv2.BORDER_REFLECT101)
        weight = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        accumulated += warped * weight
        total_weight += weight

    proposal = target_base.astype(np.float32)
    proposal += float(strength) * accumulated / np.maximum(total_weight, 1e-3)
    proposal = np.clip(proposal, 0, 255).astype(np.uint8)
    feature = make_confidence_input(target_base, proposal, target_lr, 1, ANCHORS)
    return feature, frames[1], target_lr, proposal


class VimeoMCFTStream(IterableDataset):
    def __init__(self, split, max_triplets, patches_per_triplet, patch, seed,
                 shuffle_buffer=512, flow_method="dis", proposal_strength=0.5):
        super().__init__()
        self.split = split
        self.max_triplets = max_triplets
        self.patches_per_triplet = patches_per_triplet
        self.patch = patch
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.flow_method = flow_method
        self.proposal_strength = proposal_strength

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        from datasets import load_dataset

        source = load_dataset(DATASET_ID, split=self.split, streaming=True)
        if self.split == "train" and self.shuffle_buffer:
            source = source.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        rng = np.random.default_rng(self.seed)
        for triplet_index, row in enumerate(source):
            if self.max_triplets and triplet_index >= self.max_triplets:
                break
            feature, target, lr, _ = make_triplet_example(
                (row["im1"], row["im2"], row["im3"]),
                self.flow_method, self.proposal_strength,
            )
            h, w = target.shape[:2]
            if self.patch > min(h, w):
                raise ValueError(f"patch {self.patch} exceeds streamed frame {w}x{h}")
            for _ in range(self.patches_per_triplet):
                x = int(rng.integers(0, (w-self.patch)//2+1)) * 2
                y = int(rng.integers(0, (h-self.patch)//2+1)) * 2
                fp = feature[y:y+self.patch, x:x+self.patch].astype(np.float32)
                gt = target[y:y+self.patch, x:x+self.patch]
                obs = lr[y//2:(y+self.patch)//2, x//2:(x+self.patch)//2]
                yield (
                    torch.from_numpy(fp).permute(2, 0, 1),
                    torch.from_numpy(gt.copy()).permute(2, 0, 1).float()/255.0,
                    torch.from_numpy(obs.copy()).permute(2, 0, 1).float()/255.0,
                )


def _loss(model, x, target, lr):
    base, protected = x[:, :3], x[:, 3:6]
    prediction = model(x, base)
    charb = torch.mean(torch.sqrt((prediction-target)**2+1e-6))
    grad = gradient_loss(prediction, target)
    observation = torch.mean(torch.abs(torch.nn.functional.avg_pool2d(prediction, 2)-lr))
    residual = torch.mean(torch.abs(prediction-protected))
    pred_patch = torch.mean(torch.abs(prediction-target), dim=(1, 2, 3))
    protected_patch = torch.mean(torch.abs(protected-target), dim=(1, 2, 3))
    guard = torch.mean(torch.relu(pred_patch-protected_patch))
    return charb+0.15*grad+0.20*observation+0.005*residual+0.25*guard


def _validate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for x, target, _ in loader:
            x, target = x.to(device), target.to(device)
            prediction = model(x, x[:, :3])
            total += float(torch.mean(torch.abs(prediction-target))) * len(x)
            count += len(x)
    if not count:
        raise RuntimeError("validation stream returned no samples")
    return total/count


def train(args):
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    model = build_model("confidence", args.width, args.blocks).to(device)
    if args.init:
        checkpoint = torch.load(args.init, map_location=device, weights_only=True)
        if checkpoint.get("architecture") != "confidence":
            raise ValueError("--init must be a confidence-architecture checkpoint")
        model.load_state_dict(checkpoint["state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = math.inf
    for epoch in range(args.epochs):
        train_data = VimeoMCFTStream(
            "train", args.train_triplets, args.patches_per_triplet, args.patch,
            args.seed+epoch, args.shuffle_buffer, args.flow_method, args.proposal_strength,
        )
        loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0)
        model.train()
        running = 0.0
        count = 0
        for batch_index, (x, target, lr) in enumerate(loader, 1):
            x, target, lr = x.to(device), target.to(device), lr.to(device)
            loss = _loss(model, x, target, lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach())*len(x)
            count += len(x)
            if args.log_every and batch_index % args.log_every == 0:
                expected = math.ceil(
                    args.train_triplets*args.patches_per_triplet/args.batch_size
                )
                print(
                    f"epoch {epoch+1} batch {batch_index}/{expected} "
                    f"train {running/count:.6f}", flush=True
                )
        if not count:
            raise RuntimeError("training stream returned no samples")
        validation_data = VimeoMCFTStream(
            "test", args.val_triplets, args.val_patches_per_triplet, args.patch,
            args.seed, 0, args.flow_method, args.proposal_strength,
        )
        validation = _validate(
            model, DataLoader(validation_data, batch_size=args.batch_size, num_workers=0), device
        )
        print("epoch", epoch+1, "train", running/count, "val_l1", validation)
        if validation < best:
            best = validation
            torch.save({
                "state": model.state_dict(), "width": args.width, "blocks": args.blocks,
                "architecture": "confidence", "val_l1": best,
                "pretraining_dataset": DATASET_ID,
                "pretraining_protocol": "streamed_triplet_endpoint_anchors_x2",
            }, args.output)


def smoke(args):
    data = VimeoMCFTStream(args.split, args.triplets, 1, args.patch, args.seed, 0,
                           args.flow_method, args.proposal_strength)
    count = 0
    for x, target, lr in data:
        print("sample", count+1, "feature", tuple(x.shape), "target", tuple(target.shape),
              "lr", tuple(lr.shape), "finite", bool(torch.isfinite(x).all()))
        count += 1
    if count != args.triplets:
        raise RuntimeError(f"requested {args.triplets} samples, received {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--patch", type=int, default=96)
    common.add_argument("--flow-method", choices=("farneback", "dis"), default="dis")
    common.add_argument("--proposal-strength", type=float, default=0.5)
    common.add_argument("--seed", type=int, default=7)
    check = sub.add_parser("smoke", parents=[common])
    check.add_argument("--split", choices=("train", "test"), default="test")
    check.add_argument("--triplets", type=int, default=2)
    fit = sub.add_parser("train", parents=[common])
    fit.add_argument("--output", required=True)
    fit.add_argument("--init")
    fit.add_argument("--train-triplets", type=int, default=2000)
    fit.add_argument("--val-triplets", type=int, default=200)
    fit.add_argument("--patches-per-triplet", type=int, default=2)
    fit.add_argument("--val-patches-per-triplet", type=int, default=1)
    fit.add_argument("--shuffle-buffer", type=int, default=512)
    fit.add_argument("--log-every", type=int, default=10)
    fit.add_argument("--epochs", type=int, default=4)
    fit.add_argument("--batch-size", type=int, default=8)
    fit.add_argument("--width", type=int, default=32)
    fit.add_argument("--blocks", type=int, default=6)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--device", default="auto")
    args = parser.parse_args()
    smoke(args) if args.command == "smoke" else train(args)


if __name__ == "__main__":
    main()
    # On the macOS system-Python/LibreSSL combination used by this project,
    # Hugging Face's HTTP worker can remain alive after an exhausted streaming
    # dataset. All outputs/checkpoints are synchronously closed before main()
    # returns, so flush user-visible output and avoid an indefinite interpreter
    # shutdown wait. Imported use and exception paths are unaffected.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
