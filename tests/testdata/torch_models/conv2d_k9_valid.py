import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dK9Valid(nn.Module):
    """Regular 2D conv with a 9x9 valid kernel.

    The counterpart of depthwise_conv2d_k9_valid for the non-depthwise path. A 9x9
    kernel sits in the band the high-level match used to admit (per-axis <= 9) while
    the EK lowering refuses it (per-axis <= 7), so it fell through to the hand-rolled
    Conv2DPattern body and came out wrong nearly everywhere. The match now stops at 7
    and the op is left on the host, which is what this guards.
    See synaptics-torq/torq-compiler-dev#1954.

    The fallback is a stopgap: this geometry is correct on silicon once the border is
    materialized in memory. See synaptics-torq/torq-compiler-dev#2160.

    Weights are seeded so every instantiation is identical, which keeps the torq output
    and the reference in sync when the model is loaded more than once.
    """

    def __init__(self, in_ch: int = 16, out_ch: int = 16, kernel_size: int = 9):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size, generator=g).to(torch.bfloat16)
        )
        self.bias = nn.Parameter(torch.randn(out_ch, generator=g).to(torch.bfloat16))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias)


def get_example_inputs():
    # Small frame keeps the case tiny; only the kernel size drives the test.
    g = torch.Generator().manual_seed(1)
    return (torch.randn(1, 16, 24, 24, generator=g).to(torch.bfloat16),)
