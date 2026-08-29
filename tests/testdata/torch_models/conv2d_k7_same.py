import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dK7Same(nn.Module):
    """Regular 2D conv with a 7x7 same-padded kernel.

    The widest same-padded kernel the hardware-forged halo can carry, and the control for
    conv2d_k9_same. A 7-wide kernel gives kernel_left = kernel_right = 3, exactly what the
    2-bit knl_l / knl_r CE fields hold, so it exercises the halo at its limit.

    Unlike its sibling this one is inside the match and runs on the NPU, so anything that
    changes how the halo width reaches the hardware has to keep it right.
    See synaptics-torq/torq-compiler-dev#2160.

    Weights are seeded so every instantiation is identical.
    """

    def __init__(self, in_ch: int = 16, out_ch: int = 16, kernel_size: int = 7):
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
