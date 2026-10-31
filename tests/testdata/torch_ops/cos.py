"""Cos operator fixtures built with PyTorch."""

import torch
import torch.nn as nn

KIND = "torch"


class CosModule(nn.Module):
    def forward(self, x):
        return torch.cos(x)


def make_cos():
    return CosModule().eval()


IO = {
    "cos-coarse-1024x32xbf16": [[[1024, 32], "bf16"]],
    "cos-exact-1024x32xbf16": [[[1024, 32], "bf16"]],
}

CASES = {
    "cos-coarse-1024x32xbf16": make_cos,
    "cos-exact-1024x32xbf16": make_cos,
}
