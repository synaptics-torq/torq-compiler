"""Resize operator fixtures built with torq.testing.onnx_builder."""

import numpy as np

from torq.testing.onnx_builder import BF16, F32, OnnxGraph

KIND = "onnx"


def make_resize_nearest_upsample_fpn_bf16():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=22)
    inputs = [
        g.input(f"input_{i}", BF16, [1, 16, size, size]) for i, size in enumerate([10, 20, 40, 80])
    ]
    roi = g.const(np.empty([0]), F32)
    resized = [
        g.op("Resize", x, roi, g.const([1.0, 1.0, scale, scale], F32),
             coordinate_transformation_mode="asymmetric", mode="nearest", nearest_mode="floor")
        for x, scale in zip(inputs, [8.0, 4.0, 2.0])
    ]
    concat = g.op("Concat", *resized, inputs[3], axis=1)
    weights = g.const(rng.uniform(-0.00640869140625, 0.00592041015625, [16, 64, 3, 3]), BF16)
    bias = g.const(rng.uniform(-0.337890625, 0.96484375, [16]), BF16)
    y = g.op("Conv", concat, weights, bias, dilations=[1, 1], group=1, kernel_shape=[3, 3],
             pads=[1, 1, 1, 1], strides=[1, 1], dtype=BF16, shape=[1, 16, 80, 80])
    return g.export(y)


CASES = {
    "resize_nearest_upsample_fpn_bf16": make_resize_nearest_upsample_fpn_bf16,
}

CONFIG = {
    "resize_nearest_upsample_fpn_bf16": {"comparison": {"fp_max_tol": 0.05, "allowed_wrong": 0.03}},
}
