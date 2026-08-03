import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv1dEven(nn.Module):
    """Even-kernel depthwise Conv1D with asymmetric "same" padding.

    An even "same" kernel needs asymmetric padding (total pad k-1 is odd), exercising the
    asymmetric-pad fold on the NSS path. Applied functionally so "pad + conv" is one graph:
    nn.Conv1d(padding='same') hits a separate even-kernel lowering bug, and staying
    functional avoids extracting a bare valid-conv sub-layer.
    See synaptics-torq/torq-compiler-dev#1954.

    Weights are seeded so every instantiation is identical, which keeps the torq
    output and the reference in sync when the model is loaded more than once.
    """

    def __init__(self, channels: int = 64, kernel_size: int = 8):
        super().__init__()
        self.groups = channels
        self.pad = ((kernel_size - 1) // 2, kernel_size // 2)  # asymmetric, e.g. (3, 4)
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
    return (torch.randn(1, 64, 32, generator=g).to(torch.bfloat16),)
