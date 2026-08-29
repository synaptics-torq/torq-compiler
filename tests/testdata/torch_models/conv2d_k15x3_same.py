import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dK15x3Same(nn.Module):
    """Regular 2D conv 15 high by 3 wide, past the height cap and within the width cap.

    Isolates the vertical axis: the horizontal border asks for 1 per side, well inside the
    knl_l / knl_r fields that cap conv2d_k9_same. With the height cap at 7 the match declines
    this and the host takes it. On SL2619 it computes bit-exact on the NPU, so it is the case
    that starts running there if the height cap is raised to 15, and what has to stay correct
    when it does. See synaptics-torq/torq-compiler-dev#2160.

    Weights are seeded so every instantiation is identical.
    """

    def __init__(self, in_ch: int = 16, out_ch: int = 16, kh: int = 15, kw: int = 3):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.pad = (kh // 2, kw // 2)
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kh, kw, generator=g).to(torch.bfloat16)
        )
        self.bias = nn.Parameter(torch.randn(out_ch, generator=g).to(torch.bfloat16))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=self.pad)


def get_example_inputs():
    g = torch.Generator().manual_seed(1)
    return (torch.randn(1, 16, 48, 48, generator=g).to(torch.bfloat16),)
