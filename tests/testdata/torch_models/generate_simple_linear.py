import sys
from pathlib import Path

# Import simple_linear as a module so pickle records the proper namespace
sys.path.insert(0, str(Path(__file__).parent))
from simple_linear import SimpleLinear

import torch

model = SimpleLinear()
model = model.to(torch.bfloat16)
example_inputs = torch.randn(2, 8, dtype=torch.bfloat16)
out_path = Path(__file__).parent / "simple_linear.pt"
torch.save({"model": model, "example_inputs": example_inputs}, out_path)
print(f"Saved to {out_path}")
