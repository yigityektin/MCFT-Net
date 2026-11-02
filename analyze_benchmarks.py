"""Summarize held-out REDS comparisons with paired bootstrap intervals."""
from __future__ import annotations

import csv
import random
from pathlib import Path

CLIPS = ("002", "003", "004", "005")
METHODS = ("replay", "constrained", "bicubic")
METRICS = ("psnr_y", "ssim_y", "edge_mae", "temporal_delta_mae")
ROOT = Path("benchmarks")


def read_clip(clip: str):
    rows = list(csv.DictReader((ROOT / f"reds_{clip}_heldout_comparison.csv").open()))
    return {
        method: {
            int(row["frame"]): row
            for row in rows
            if row["method"] == method and row["is_anchor"] == "0"
        }
        for method in METHODS
    }


def mean(values):
    return sum(values) / len(values)


def bootstrap(values, seed=7, n=10000):
    rng = random.Random(seed)
    estimates = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    return mean(values), estimates[int(0.025 * n)], estimates[int(0.975 * n)]


def main():
    clips = {clip: read_clip(clip) for clip in CLIPS}
    summary_rows = []
    interval_rows = []
    for method in METHODS:
        for metric in METRICS:
            values = [
                float(row[metric])
                for clip in CLIPS
                for row in clips[clip][method].values()
                if row[metric].lower() != "nan"
            ]
            summary_rows.append({"method": method, "metric": metric, "mean": mean(values), "frames": len(values)})
    for baseline in ("constrained", "bicubic"):
        for metric in METRICS:
            deltas = []
            for clip in CLIPS:
                common = sorted(set(clips[clip]["replay"]) & set(clips[clip][baseline]))
                for frame in common:
                    a = clips[clip]["replay"][frame][metric]
                    b = clips[clip][baseline][frame][metric]
                    if a.lower() != "nan" and b.lower() != "nan":
                        # Positive means replay is better; errors are inverted.
                        sign = 1 if metric in ("psnr_y", "ssim_y") else -1
                        deltas.append(sign * (float(a) - float(b)))
            m, lo, hi = bootstrap(deltas)
            interval_rows.append({"comparison": f"replay-vs-{baseline}", "metric": metric, "delta": m, "ci95_low": lo, "ci95_high": hi, "frames": len(deltas)})
    with (ROOT / "heldout_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("method", "metric", "mean", "frames"))
        writer.writeheader(); writer.writerows(summary_rows)
    with (ROOT / "heldout_bootstrap.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("comparison", "metric", "delta", "ci95_low", "ci95_high", "frames"))
        writer.writeheader(); writer.writerows(interval_rows)
    for row in summary_rows:
        print(f"{row['method']:11s} {row['metric']:22s} {row['mean']:.6f}")
    print("\nPaired bootstrap (positive favors replay):")
    for row in interval_rows:
        print(f"{row['comparison']:24s} {row['metric']:22s} {row['delta']:.6f} [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]")


if __name__ == "__main__":
    main()
