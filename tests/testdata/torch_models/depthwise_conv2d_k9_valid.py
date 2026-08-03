import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv2dK9Valid(nn.Module):
    """Genuine 2D depthwise with a 9x9 valid kernel.

    A 9x9 kernel sits in the band that the high-level depthwise match used to admit
    (per-axis <= 9) but the EK lowering refuses (per-axis <= 7), so it fell through to
    the hand-rolled DWPattern body and came out wrong nearly everywhere. The match now
    stops at 7 and the op is left on the host, which is what this guards: re-widening
    the match, or teaching EK the band, has to keep this case numerically right.
    See synaptics-torq/torq-compiler-dev#1954.

    The fallback is a stopgap: this geometry is correct on silicon once the border is
    materialized in memory. See synaptics-torq/torq-compiler-dev#2160.

    Weights are seeded so every instantiation is identical, which keeps the torq output
    and the reference in sync when the model is loaded more than once.
    """

    def __init__(self, channels: int = 32, kernel_size: int = 9):
        super().__init__()
        self.groups = channels
        g = torch.Generator().manual_seed(0)
        self.weight = nn.Parameter(
            torch.randn(channels, 1, kernel_size, kernel_size, generator=g).to(torch.bfloat16)
        )
        self.bias = nn.Parameter(torch.randn(channels, generator=g).to(torch.bfloat16))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, groups=self.groups)


def get_example_inputs():
    # Small frame keeps the case tiny; only the kernel size drives the test.
    g = torch.Generator().manual_seed(1)
    return (torch.randn(1, 32, 24, 24, generator=g).to(torch.bfloat16),)
