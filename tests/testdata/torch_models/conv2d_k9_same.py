import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dK9Same(nn.Module):
    """Regular 2D conv with a 9x9 same-padded kernel.

    The counterpart of conv2d_k9_valid, and the case that is genuinely broken above 7.
    Same-padding leaves the border zeros to the hardware-forged halo, whose width travels
    in the 2-bit knl_l / knl_r CE fields: a 9-wide kernel asks for 4 per side and the
    field holds 3. Wrong on silicon and on the simulator, unlike conv2d_k7_same.

    The match stops at 7, so this is left on the host and passes that way. It guards the
    distinction: the valid sibling may be re-admitted, this one may not.
    See synaptics-torq/torq-compiler-dev#2160.

    Weights are seeded so every instantiation is identical.
    """

    def __init__(self, in_ch: int = 16, out_ch: int = 16, kernel_size: int = 9):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.pad = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size, generator=g).to(torch.bfloat16)
        )
        self.bias = nn.Parameter(torch.randn(out_ch, generator=g).to(torch.bfloat16))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=self.pad)


def get_example_inputs():
    g = torch.Generator().manual_seed(1)
    return (torch.randn(1, 16, 24, 24, generator=g).to(torch.bfloat16),)
