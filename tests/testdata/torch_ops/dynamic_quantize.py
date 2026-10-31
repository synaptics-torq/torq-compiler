"""Dynamic quantize operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import F32, OnnxGraph, typed

KIND = "onnx"


def _make_dynamic_quantize(input_shape):
    g = OnnxGraph(opset=22)
    x = g.input("x", F32, input_shape)
    y, _, _ = g.op("DynamicQuantizeLinear", x, outputs=3)
    return g.export(typed(y, np.uint8, input_shape))


CASES = {
    "dynamicquantize": partial(_make_dynamic_quantize, [1, 1, 288]),
    "dynamicquantize_large": partial(_make_dynamic_quantize, [1, 1, 131072]),
}

# ONNX DynamicQuantizeLinear tensors are always f32, so DQL is only reachable in a bf16 pipeline
# with --torq-convert-dtypes.
CONFIG = {
    "dynamicquantize": {
        "compiler_options": [
            "--torq-disable-host", "--torq-disable-css", "--torq-convert-dtypes",
            "--torq-convert-io-dtype",
        ],
        "comparison": {"int_tol": 2.0},
    },
    "dynamicquantize_large": {
        "compiler_options": [
            "--torq-disable-host", "--torq-disable-css", "--torq-convert-dtypes",
            "--torq-convert-io-dtype",
        ],
        "comparison": {"int_tol": 2.0},
    },
}
