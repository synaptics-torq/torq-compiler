"""Gelu operator fixtures built with torq.testing.onnx_builder."""

import numpy as np

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def make_gelu_gemma():
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, [1, 1024, 64])
    return g.export(g.op("Gelu", x, approximate="tanh", dtype=BF16, shape=[1, 1024, 64]))


def make_erf_mlp_dup_lram_load_bf16():
    """PP-OCR MLP regression for duplicate GELU operand loads in LRAM.

    At [1,192,3,608], two identical loads need 705,544 B; merging them needs
    472,072 B and fits the 523,008 B LRAM. Equal-index frees must precede
    allocations in the interval sweep or the merge can be rejected.
    Preserve the ONNX Div/Erf/Add/Mul/Mul layout and bf16-exact splat weights:
    torch.export changes the layout and does not reproduce this failure.
    """
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, [1, 96, 3, 608])
    hidden = {"dtype": BF16, "shape": [1, 192, 3, 608]}

    def conv1x1(x, weight, bias, out_channels, **kwargs):
        weights = g.const(np.full([out_channels, x.shape[1], 1, 1], weight), BF16)
        return g.op("Conv", x, weights, g.const(np.full([out_channels], bias), BF16),
                    dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                    strides=[1, 1], **kwargs)

    convolved = conv1x1(x, 0.0625, 0.125, 192, **hidden)
    erf = g.op("Erf", g.op("Div", convolved, g.const(1.4140625, BF16), **hidden), **hidden)
    shifted = g.op("Add", erf, g.const(1.0, BF16), **hidden)
    product = g.op("Mul", convolved, shifted, **hidden)
    gelu = g.op("Mul", product, g.const(0.5, BF16), **hidden)
    return g.export(conv1x1(gelu, -0.03125, 0.09375, 160, dtype=BF16, shape=[1, 160, 3, 608]))


CASES = {
    "erf_mlp_dup_lram_load_bf16": make_erf_mlp_dup_lram_load_bf16,
    "gelu-gemma": make_gelu_gemma,
}

CONFIG = {
    # LRAM-limit regression pinned to the no-slicing path used for real models (the sliced path
    # fails this shape for an unrelated reason: an LRAM fragmentation failure).
    "erf_mlp_dup_lram_load_bf16": {
        "compiler_options": [
            "--torq-disable-slicing", "--torq-disable-host", "--torq-disable-css",
        ],
        "comparison": {
            "fp_max_tol": 0.2, "fp_avg_tol": 0.03,
            "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.001,
        },
    },
    # GELU can produce tiny values in the far negative tail, where the true result is already
    # effectively zero. Observed: expected -1.611188e-7, observed -0.0.
    "gelu-gemma": {"comparison": {"epsilon": 2e-5}, "reference": "numpy_gelu_reference_results"},
}
