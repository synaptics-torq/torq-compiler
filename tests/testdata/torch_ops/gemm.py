"""Gemm operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, F32, I8, I32, OnnxGraph

KIND = "onnx"


def _make_gemm(input_shape, weight_shape, *, weight_range=None, bias_range=None):
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, input_shape)
    bias_shape = [weight_shape[1]]
    if weight_range is None:
        weights = np.full(weight_shape, 0.010009765625)
    else:
        weights = rng.uniform(*weight_range, weight_shape)
    bias = np.zeros(bias_shape) if bias_range is None else rng.uniform(*bias_range, bias_shape)
    y = g.op("Gemm", x, g.const(weights, BF16), g.const(bias, BF16), alpha=1.0, beta=1.0,
             transA=0, transB=0, dtype=BF16, shape=[input_shape[0], weight_shape[1]])
    return g.export(y)


def make_qdq_gemm():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=18)
    x = g.input("x", I8, [1, 512])
    weights = g.const(rng.integers(-98, 128, [1000, 512], dtype=np.int8), I8)
    bias = g.const(rng.integers(-20485, -11013, [1000], dtype=np.int32), I32)
    y = g.op("Gemm", g.dequantize(x, 0.00784226506948471, -1),
             g.dequantize(weights, 0.028673745691776276, 0),
             g.dequantize(bias, g.const([0.00022486710804514587], F32), 0, zp_dtype=I32),
             alpha=1.0, beta=1.0, transB=1)
    return g.export(g.quantize(y, 0.11942443996667862, 44, dtype=I8, shape=[1, 1000]))


CASES = {
    "gemm-fc-row-channel-tiling-bf16": partial(_make_gemm, [207, 1152], [1152, 288]),
    "gemm_bf16": partial(
        _make_gemm,
        [207, 1152],
        [1152, 288],
        weight_range=(-3.671875, 3.546875),
        bias_range=(-0.94921875, 0.859375),
    ),
    'qdq-gemm': make_qdq_gemm,
}

CONFIG = {
    "gemm-fc-row-channel-tiling-bf16": {"comparison": {"fp_max_tol": 0.05}},
    "gemm_bf16": {"comparison": {"fp_max_tol": 0.015}},
}
