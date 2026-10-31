"""Rnn operator fixtures built with torq.testing.onnx_builder."""

import numpy as np

from torq.testing.onnx_builder import BF16, OnnxGraph, typed

KIND = "onnx"


def make_rnn_s4_b1_h8_bf16():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=17)
    x = g.input("input_0", BF16, [4, 1, 8])
    h0 = g.input("input_1", BF16, [1, 1, 8])
    w = g.const(rng.uniform(-0.337890625, 0.3515625, [1, 8, 8]), BF16)
    r = g.const(rng.uniform(-0.353515625, 0.33984375, [1, 8, 8]), BF16)
    b = g.const(rng.uniform(-0.2890625, 0.3359375, [1, 16]), BF16)
    y, y_h = g.op("RNN", x, w, r, b, None, h0, outputs=2, activations=["Tanh"], hidden_size=8)
    return g.export(typed(y, BF16, [4, 1, 1, 8]), typed(y_h, BF16, [1, 1, 8]))


CASES = {
    "rnn_s4_b1_h8_bf16": make_rnn_s4_b1_h8_bf16,
}
