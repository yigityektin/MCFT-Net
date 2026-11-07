"""Structural smoke test, with optional checkpoint/model-forward validation."""
import ast
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Optional MCFT-Net checkpoint for a real forward test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    tree = ast.parse(Path("mcft_net.py").read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name == "train" for n in tree.body)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "infer" for n in tree.body)
    compile(Path("mcft_net.py").read_text(), "mcft_net.py", "exec")
    print("MCFT-Net source smoke test passed")
    if args.model:
        import torch
        import mcft_net

        device = torch.device(args.device)
        checkpoint = torch.load(args.model, map_location=device, weights_only=True)
        architecture = checkpoint.get("architecture", "legacy")
        model = mcft_net.build_model(
            architecture,
            checkpoint["width"], checkpoint["blocks"]
        ).to(device).eval()
        model.load_state_dict(checkpoint["state"])
        input_channels = 11 if architecture == "confidence" else 8
        x = torch.zeros(1, input_channels, 32, 32, device=device)
        with torch.inference_mode():
            y = model(x, x[:, :3])
        assert tuple(y.shape) == (1, 3, 32, 32), tuple(y.shape)
        print("MCFT-Net checkpoint load and forward test passed")


if __name__ == "__main__":
    main()
