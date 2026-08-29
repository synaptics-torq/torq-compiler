import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv2dK15x3Same(nn.Module):
    """Genuine 2D depthwise 15 high by 3 wide, the depthwise counterpart of conv2d_k15x3_same.

    Both spatial extents are larger than 1, so this is not a 1D depthwise and does not take
    lowerDw1dStride1ToHw, which carries large kernels by avoiding the halo entirely. Declined at
    the current height cap; bit-exact on SL2619 if the cap is raised to 15.
    See synaptics-torq/torq-compiler-dev#2160.

    Weights are seeded so every instantiation is identical.
    """

    def __init__(self, channels: int = 32, kh: int = 15, kw: int = 3):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.groups = channels
        self.pad = (kh // 2, kw // 2)
        self.weight = nn.Parameter(
            torch.randn(channels, 1, kh, kw, generator=g).to(torch.bfloat16)
        )
        self.bias = nn.Parameter(torch.randn(channels, generator=g).to(torch.bfloat16))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=self.pad, groups=self.groups)


def get_example_inputs():
    g = torch.Generator().manual_seed(1)
    return (torch.randn(1, 32, 48, 48, generator=g).to(torch.bfloat16),)
