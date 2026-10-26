"""Print architecture, parameter count, and training metadata for an MCFT checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mcft_net import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "unknown")
    width = checkpoint.get("width")
    blocks = checkpoint.get("blocks")
    print("checkpoint:", args.checkpoint)
    print("architecture:", architecture)
    print("width:", width)
    print("blocks:", blocks)
    for key in ("training", "pairs_only_steps", "val_l1", "teacher"):
        if key in checkpoint:
            print(f"{key}:", checkpoint[key])
    if width is not None and blocks is not None and architecture != "unknown":
        model = build_model(architecture, int(width), int(blocks))
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print("parameters:", total)
        print("trainable_parameters:", trainable)


if __name__ == "__main__":
    main()
