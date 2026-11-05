"""CPU-only structural smoke test for the reproducibility checklist."""
import ast
from pathlib import Path

def main():
    tree = ast.parse(Path("mcft_net.py").read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name == "train" for n in tree.body)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "infer" for n in tree.body)
    compile(Path("mcft_net.py").read_text(), "mcft_net.py", "exec")
    print("MCFT-Net source smoke test passed")


if __name__ == "__main__":
    main()
