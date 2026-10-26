"""Aggregate non-anchor quality from per-clip evaluation CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

METRICS = ("psnr_y", "ssim_y", "edge_mae", "temporal_delta_mae")


def read(path: Path, label: str):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("is_anchor") == "0"]
    if not rows:
        raise ValueError(f"no non-anchor rows in {path}")
    result = {"clip": label, "frames": len(rows)}
    for metric in METRICS:
        values = []
        for row in rows:
            value = row.get(metric, "nan")
            if value.lower() != "nan":
                values.append(float(value))
        result[metric] = sum(values) / len(values) if values else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, metavar="CLIP=CSV")
    parser.add_argument("--output", type=Path, default=Path("reds_aggregate_quality.csv"))
    args = parser.parse_args()
    rows = []
    for item in args.input:
        if "=" not in item:
            parser.error(f"expected CLIP=CSV, got {item!r}")
        label, filename = item.split("=", 1)
        rows.append(read(Path(filename), label))
    average = {"clip": "AVERAGE", "frames": sum(row["frames"] for row in rows)}
    for metric in METRICS:
        average[metric] = sum(row[metric] * row["frames"] for row in rows) / average["frames"]
    rows.append(average)
    args.output.write_text(
        "\n".join(",".join(str(row[key]) for key in ("clip", "frames", *METRICS)) for row in [
            {key: key for key in ("clip", "frames", *METRICS)}, *rows
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {args.output}")
    print("average non-anchor:", "PSNR-Y", average["psnr_y"], "SSIM-Y", average["ssim_y"],
          "Edge-MAE", average["edge_mae"], "Temporal-MAE", average["temporal_delta_mae"])


if __name__ == "__main__":
    main()
