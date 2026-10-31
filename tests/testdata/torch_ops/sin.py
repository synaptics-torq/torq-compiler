"""Sin operator fixtures built with PyTorch."""

import torch
import torch.nn as nn

KIND = "torch"


class SinModule(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def make_sin():
    return SinModule().eval()


IO = {
    "sin-coarse-1024x32xbf16": [[[1024, 32], "bf16"]],
    "sin-exact-1024x32xbf16": [[[1024, 32], "bf16"]],
}

CASES = {
    "sin-coarse-1024x32xbf16": make_sin,
    "sin-exact-1024x32xbf16": make_sin,
}

# sin is optimized for RoPE, which only runs on (often small) positive inputs: negative inputs
# are not quite as accurate.
CONFIG = {
    "sin-coarse-1024x32xbf16": {"input_range": [0, 12], "comparison": {"allowed_wrong": 0.06}},
    "sin-exact-1024x32xbf16": {"input_range": [0, 3]},
}
