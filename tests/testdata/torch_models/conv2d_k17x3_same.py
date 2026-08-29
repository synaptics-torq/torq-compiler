import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dK17x3Same(nn.Module):
    """Regular 2D conv 17 high by 3 wide, the first height the hardware gets wrong.

    The sibling of conv2d_k15x3_same: on SL2619 the border comes out wrong on 63% of the output
    with no diagnostic, while the simulator computes it correctly. The match declines it, so it
    is left on the host and passes that way. It is what fails if the height cap is ever widened
    to 17 or beyond on simulator evidence. See synaptics-torq/torq-compiler-dev#2160.

    Weights are seeded so every instantiation is identical.
    """

    def __init__(self, in_ch: int = 16, out_ch: int = 16, kh: int = 17, kw: int = 3):
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
