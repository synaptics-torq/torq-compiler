import numpy as np
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from torch_mlir import torchscript

# Note: compile torchscript 
# please refer to 
# https://github.com/syna-astra-dev/iree-synaptics-synpu/blob/main/doc/model_conversion.md#convert-torch-model-to-mlir
# to compile the torchscript or pip install torch_mlir

# about iree and torch-mlir
# iree include torch-mlir files to compile into iree binary, torch-mlir related python modules not compiled by iree

# Define a simple neural network with a Conv2D layer
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class MatMulModel(nn.Module):
    def __init__(self):
        super(MatMulModel, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


def model_conv():
    model = SimpleConvNet()
    model = model.to(torch.bfloat16)
    input_tensor = torch.randn(1, 1, 28, 28).to(torch.bfloat16)
    return model, [input_tensor]

def model_matmul():
    model = MatMulModel()
    model = model.to(torch.bfloat16)
    x = torch.randn(2, 3).to(torch.bfloat16)
    y = torch.randn(3, 2).to(torch.bfloat16)
    return model, [x, y]


def gen_mlir(model_name, model, input_tensors):
    print(f"Generating MLIR for model: {model_name}")
    def compile(model, input_tensor, output_type):
        module = torchscript.compile(model,
                                input_tensors,
                                output_type,
                                use_tracing=False,
                                verbose=False)

        output = "torch.mlir"
        if output_type == torchscript.OutputType.STABLEHLO:
            output = "stablehlo.mlir"
        elif output_type == torchscript.OutputType.LINALG_ON_TENSORS:
            output = "linalg.mlir"
        
        out_dir = Path("out")
        out_dir.mkdir(exist_ok=True, parents=True)

        output_file = out_dir / f"{model_name}.{output}"
        
        with open(output_file, "w") as fp:
            fp.write(module.operation.get_asm())
    
    compile(model, input_tensors, torchscript.OutputType.TORCH)
    compile(model, input_tensors, torchscript.OutputType.LINALG_ON_TENSORS)


model_catalog = {
    "conv": model_conv,
    "matmul": model_matmul
}

def main():
    """
    Generate torch.mlir and linalg.mlir files under out directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="add", choices=model_catalog.keys(), help="Model type")
    args = parser.parse_args()

    torch_model, input_tensors = model_catalog[args.model]()
    gen_mlir(args.model, torch_model, input_tensors)
    

if __name__ == "__main__":
    main()