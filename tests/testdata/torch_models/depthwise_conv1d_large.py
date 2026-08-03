import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv1dLarge(nn.Module):
    """Large-kernel depthwise Conv1D with symmetric "same" padding.

    A kernel with kernel_top = (k-1)//2 > 7 exceeds the DW HW's implicit
    boundary-zero halo, so the "same" pad must be kept explicit as a valid conv
    on the NSS path rather than folded into the conv (otherwise the frame edges
    miscompute on silicon). Applied functionally so "pad + conv" is one graph and
    no bare valid-conv sub-layer is extracted.
    See synaptics-torq/torq-compiler-dev#1954 sub-bug #2.

    Weights are seeded so every instantiation is identical, which keeps the torq
    output and the reference in sync when the model is loaded more than once.
    """

    def __init__(self, channels: int = 32, kernel_size: int = 31):
        super().__init__()
        self.groups = channels
        pad = (kernel_size - 1) // 2  # symmetric for an odd kernel
        self.pad = (pad, pad)
        g = torch.Generator().manual_seed(0)
        self.weight = nn.Parameter(
            torch.randn(channels, 1, kernel_size, generator=g).to(torch.bfloat16)
        )
        self.bias = nn.Parameter(torch.randn(channels, generator=g).to(torch.bfloat16))

    def forward(self, x):
        return F.conv1d(F.pad(x, self.pad), self.weight, self.bias, groups=self.groups)


def get_example_inputs():
    # Small width keeps the case tiny; only the shape drives the test.
    g = torch.Generator().manual_seed(1)
    return (torch.randn(1, 32, 64, generator=g).to(torch.bfloat16),)
