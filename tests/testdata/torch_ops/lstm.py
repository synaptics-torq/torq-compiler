"""Lstm operator fixtures built with torq.testing.onnx_builder."""

import numpy as np

from torq.testing.onnx_builder import BF16, OnnxGraph, typed

KIND = "onnx"


def make_lstm_s4_b1_h8_bf16():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=17)
    x = g.input("input_0", BF16, [4, 1, 8])
    h0 = g.input("input_1", BF16, [1, 1, 8])
    c0 = g.input("input_2", BF16, [1, 1, 8])
    w = g.const(rng.uniform(-0.353515625, 0.3515625, [1, 32, 8]), BF16)
    r = g.const(rng.uniform(-0.349609375, 0.353515625, [1, 32, 8]), BF16)
    b = g.const(rng.uniform(-0.34375, 0.353515625, [1, 64]), BF16)
    y, y_h, y_c = g.op("LSTM", x, w, r, b, None, h0, c0, outputs=3, hidden_size=8)
    return g.export(
        typed(y, BF16, [4, 1, 1, 8]), typed(y_h, BF16, [1, 1, 8]), typed(y_c, BF16, [1, 1, 8])
    )


CASES = {
    "lstm_s4_b1_h8_bf16": make_lstm_s4_b1_h8_bf16,
}

CONFIG = {
    "lstm_s4_b1_h8_bf16": {"comparison": {"fp_max_tol": 0.035}},
}
