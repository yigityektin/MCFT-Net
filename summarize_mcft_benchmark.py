"""Aggregate per-frame MCFT benchmark CSV files into a clip summary."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = ("psnr_y", "ssim_y", "edge_mae", "temporal_delta_mae")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_metric(rows: list[dict[str, str]], metric: str) -> float:
    values = [float(row[metric]) for row in rows if row[metric].lower() != "nan"]
    return float(np.mean(values)) if values else float("nan")


def summarize_clip(root: Path, clip: str, csv_name: str) -> list[dict[str, str | float]]:
    path = root / clip / csv_name
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_rows(path)
    methods = sorted({row["method"] for row in rows})
    result: list[dict[str, str | float]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        nonanchor_rows = [row for row in method_rows if int(row["is_anchor"]) == 0]
        for split, split_rows in (("all", method_rows), ("nonanchor", nonanchor_rows)):
            record: dict[str, str | float] = {
                "clip": clip,
                "split": split,
                "method": method,
                "frames": len(split_rows),
            }
            for metric in METRICS:
                record[metric] = mean_metric(split_rows, metric)
            result.append(record)
    return result


def add_gains(rows: list[dict[str, str | float]], baseline: str) -> None:
    by_clip_split = {}
    for row in rows:
        by_clip_split.setdefault((row["clip"], row["split"]), {})[row["method"]] = row
    for methods in by_clip_split.values():
        base = methods.get(baseline)
        if not base:
            continue
        for row in methods.values():
            for metric in METRICS:
                value = float(row[metric])
                base_value = float(base[metric])
                if metric in ("edge_mae", "temporal_delta_mae"):
                    row[f"{metric}_gain_vs_{baseline}"] = base_value - value
                else:
                    row[f"{metric}_gain_vs_{baseline}"] = value - base_value


def average_rows(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["split"], row["method"]), []).append(row)
    averages: list[dict[str, str | float]] = []
    numeric_keys = [key for key in rows[0] if key not in ("clip", "split", "method")]
    for (split, method), group in sorted(grouped.items()):
        record: dict[str, str | float] = {"clip": "AVERAGE", "split": split, "method": method}
        for key in numeric_keys:
            record[key] = float(np.nanmean([float(row.get(key, "nan")) for row in group]))
        averages.append(record)
    return averages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("external_benchmarks/REDS-Val4"))
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--csv-name", default="confidence_mixed_replay_ssim_500_texture_independent_quality.csv")
    parser.add_argument("--baseline", default="mcft_v3")
    parser.add_argument("--output", type=Path, default=Path("external_benchmarks/REDS-Val4/mcft_best_texture_summary.csv"))
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str | float]] = []
    for clip in args.clips:
        rows.extend(summarize_clip(args.root, clip, args.csv_name))
    add_gains(rows, args.baseline)
    rows.extend(average_rows(rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {args.output}")
    for row in rows:
        if row["clip"] == "AVERAGE" and row["split"] == "nonanchor":
            print(
                row["method"],
                "PSNR-Y",
                row["psnr_y"],
                "SSIM-Y",
                row["ssim_y"],
                "Edge-MAE",
                row["edge_mae"],
                "Temporal-delta-MAE",
                row["temporal_delta_mae"],
            )
    if args.markdown_output:
        averages = [row for row in rows if row["clip"] == "AVERAGE" and row["split"] == "nonanchor"]
        args.markdown_output.write_text(markdown_report(averages, args.baseline), encoding="utf-8")
        print(f"[OK] wrote {args.markdown_output}")


def markdown_report(rows: list[dict[str, str | float]], baseline: str) -> str:
    ordered = sorted(rows, key=lambda row: str(row["method"]))
    lines = [
        "# MCFT Benchmark Summary",
        "",
        "Non-anchor average from the frozen benchmark CSV.",
        "",
        "| Method | PSNR-Y | SSIM-Y | Edge MAE | Temporal MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['method']} | {float(row['psnr_y']):.4f} | "
            f"{float(row['ssim_y']):.5f} | {float(row['edge_mae']):.3f} | "
            f"{float(row['temporal_delta_mae']):.4f} |"
        )
    best = next((row for row in rows if row["method"] == "best_texture"), None)
    base = next((row for row in rows if row["method"] == baseline), None)
    bicubic = next((row for row in rows if row["method"] == "bicubic"), None)
    lines.extend(["", "## Gains", ""])
    if best and bicubic:
        lines.append(
            f"Best texture vs bicubic: +{float(best['psnr_y']) - float(bicubic['psnr_y']):.3f} dB PSNR, "
            f"+{float(best['ssim_y']) - float(bicubic['ssim_y']):.5f} SSIM, "
            f"{float(best['edge_mae']) - float(bicubic['edge_mae']):.3f} Edge-MAE, "
            f"{float(best['temporal_delta_mae']) - float(bicubic['temporal_delta_mae']):.4f} Temporal-MAE."
        )
    if best and base:
        lines.append(
            f"Best texture vs {baseline}: +{float(best['psnr_y']) - float(base['psnr_y']):.3f} dB PSNR, "
            f"+{float(best['ssim_y']) - float(base['ssim_y']):.5f} SSIM, "
            f"{float(best['edge_mae']) - float(base['edge_mae']):.3f} Edge-MAE, "
            f"{float(best['temporal_delta_mae']) - float(base['temporal_delta_mae']):.4f} Temporal-MAE."
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
