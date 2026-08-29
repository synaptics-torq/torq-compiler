import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dK15x3W384(nn.Module):
    """Tall same-padded conv wide enough that tile-and-fuse has to split H.

    A tile whose first row falls inside the top border band, or whose last row falls
    inside the bottom one, needs a partial pad_top/pad_bottom. The hardware expresses
    only 0 or the full border, so the pass that materializes the border in memory has
    to admit this kernel height, and its gates are per-axis for that reason. Dormant
    while the height cap is 7: the match declines this and the host takes it.

    The value scheme keeps the comparison meaningful. Inputs and weights are 0 or 1
    and only four input channels carry weight, so every partial sum is an integer
    below 256 and therefore exact in bf16 whatever order the hardware accumulates in.
    All sixteen channels still move through LRAM, so the tiling pressure is unchanged.
    Float data here fails on rounding alone: the comparison gate is a bare relative
    one and its threshold is about two bf16 ULP.

    If the height cap is raised to 15, this is the case that aborts if the pass gates lag
    behind. To check that, raise the cap, put the pass gates back to 7 and confirm
    the abort. Do not read the pad attributes on the op instead: they can show only
    0 and the full border and the case still aborts, because the partial pad appears
    further down, in the per-slice runtime config.
    """

    LIVE_CHANNELS = 4

    def __init__(self, in_ch: int = 16, out_ch: int = 16, kh: int = 15, kw: int = 3):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.pad = (kh // 2, kw // 2)
        w = torch.randint(0, 2, (out_ch, in_ch, kh, kw), generator=g)
        w[:, self.LIVE_CHANNELS :] = 0
        self.weight = nn.Parameter(w.to(torch.bfloat16))
        self.bias = nn.Parameter(torch.randint(0, 3, (out_ch,), generator=g).to(torch.bfloat16))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=self.pad)


def get_example_inputs():
    g = torch.Generator().manual_seed(1)
    return (torch.randint(0, 2, (1, 16, 48, 384), generator=g).to(torch.bfloat16),)
