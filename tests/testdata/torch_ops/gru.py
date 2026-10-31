"""Gru operator fixtures built with torq.testing.onnx_builder."""

import numpy as np

from torq.testing.onnx_builder import BF16, OnnxGraph, typed

KIND = "onnx"


def make_gru_s4_b1_h8_bf16():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=17)
    x = g.input("input_0", BF16, [4, 1, 8])
    h0 = g.input("input_1", BF16, [1, 1, 8])
    w = g.const(rng.uniform(-0.353515625, 0.3515625, [1, 24, 8]), BF16)
    r = g.const(rng.uniform(-0.349609375, 0.353515625, [1, 24, 8]), BF16)
    b = g.const(rng.uniform(-0.322265625, 0.353515625, [1, 48]), BF16)
    y, y_h = g.op("GRU", x, w, r, b, None, h0, outputs=2, hidden_size=8, linear_before_reset=1)
    return g.export(typed(y, BF16, [4, 1, 1, 8]), typed(y_h, BF16, [1, 1, 8]))


CASES = {
    "gru_s4_b1_h8_bf16": make_gru_s4_b1_h8_bf16,
}

CONFIG = {
    # Since df56d91e the epilogue fusion of the matmuls stops at runtime data such as the
    # carried hidden state: one of the 32 outputs is one bf16 step off (23.25 for 23.375).
    "gru_s4_b1_h8_bf16": {"comparison": {"fp_max_tol": 0.035, "allowed_wrong": 0.05}},
}
