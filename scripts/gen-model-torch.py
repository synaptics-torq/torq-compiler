#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from iree.compiler.extras.fx_importer import FxImporter
from iree.compiler import ir, passmanager

"""
Generate MLIR from PyTorch models using IREE's bundled FX importer.

This replaces the previous dependency on `torch_mlir.torchscript` with
`iree.compiler.extras.fx_importer`, which is already built as part of IREE.

Outputs:
- out/<model>.torch.mlir   : Torch dialect MLIR
- out/<model>.linalg.mlir  : Linalg-on-tensors MLIR (via torch-mlir passes)
"""


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class MatMulModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


def model_conv():
    model = SimpleConvNet()
    model = model.to(torch.bfloat16)
    input_tensor = torch.randn(1, 1, 28, 28).to(torch.bfloat16)
    return model, (input_tensor,)


def model_matmul():
    model = MatMulModel()
    model = model.to(torch.bfloat16)
    x = torch.randn(2, 3).to(torch.bfloat16)
    y = torch.randn(3, 2).to(torch.bfloat16)
    return model, (x, y)


def export_to_torch_mlir(model, example_args):
    """Export model to Torch dialect MLIR using FX importer."""
    model.eval()
    prog = torch.export.export(model, example_args)

    context = ir.Context()
    fx_imp = FxImporter(context=context)
    fx_imp.import_frozen_program(prog, func_name="main")
    return str(fx_imp.module)


def lower_to_linalg(torch_mlir_str: str) -> str:
    """Lower Torch dialect MLIR to Linalg-on-tensors using built-in passes."""
    context = ir.Context()
    module = ir.Module.parse(torch_mlir_str, context=context)

    # Torch dialect -> Torch backend dialect
    pm = passmanager.PassManager.parse(
        "builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        context=context,
    )
    pm.run(module.operation)

    # Torch backend dialect -> Linalg
    pm2 = passmanager.PassManager.parse(
        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        context=context,
    )
    pm2.run(module.operation)

    return str(module)


def gen_mlir(model_name, model, example_args):
    print(f"Generating MLIR for model: {model_name}")

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Generate Torch dialect MLIR
    torch_mlir = export_to_torch_mlir(model, example_args)
    torch_file = out_dir / f"{model_name}.torch.mlir"
    torch_file.write_text(torch_mlir)
    print(f"  Wrote {torch_file}")

    # 2. Generate Linalg MLIR
    linalg_mlir = lower_to_linalg(torch_mlir)
    linalg_file = out_dir / f"{model_name}.linalg.mlir"
    linalg_file.write_text(linalg_mlir)
    print(f"  Wrote {linalg_file}")


model_catalog = {
    "conv": model_conv,
    "matmul": model_matmul,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate Torch/Linalg MLIR from PyTorch models"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="conv",
        choices=model_catalog.keys(),
        help="Model type",
    )
    args = parser.parse_args()

    torch_model, input_args = model_catalog[args.model]()
    gen_mlir(args.model, torch_model, input_args)


if __name__ == "__main__":
    main()
