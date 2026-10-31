"""Abs operator fixtures built with PyTorch."""

import torch
import torch.nn as nn

KIND = "torch"


class SimpleModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)


def make_abs():
    return SimpleModule().eval()


IO = {
    "abs": [[[2, 16], "f32"]],
}

CASES = {
    "abs": make_abs,
}
