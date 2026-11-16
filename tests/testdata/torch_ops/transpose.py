"""Transpose operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def _make_transpose(input_shape, perm):
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, input_shape)
    shape = [input_shape[i] for i in perm]
    return g.export(g.op("Transpose", x, perm=perm, dtype=BF16, shape=shape))


CASES = {
    "transpose-1x1x8192-bf16": partial(_make_transpose, [1, 1, 8192], [1, 2, 0]),
    "transpose-1x512x16x1-bf16": partial(_make_transpose, [1, 512, 16, 1], [3, 0, 1, 2]),
    "transpose-1x64x1x1-bf16": partial(_make_transpose, [1, 64, 1, 1], [0, 3, 2, 1]),
    # Ported from tests/testdata/tosa_ops, with the names of the MLIR files
    "transpose-02341-2x64x4x8x3xbf16": partial(_make_transpose, [2, 64, 4, 8, 3], [0, 2, 3, 4, 1]),
    "transpose-120-64x32x33xbf16": partial(_make_transpose, [64, 32, 33], [1, 2, 0]),
    "transpose-201-33x40x64xbf16": partial(_make_transpose, [33, 40, 64], [2, 0, 1]),
    "transpose-95x159xbf16": partial(_make_transpose, [95, 159], [1, 0]),
    "transpose-nchw2nhwc-1x64x32x32xbf16": partial(_make_transpose, [1, 64, 32, 32], [0, 2, 3, 1]),
    "transpose-nchw2nhwc-2x48x7x9xbf16": partial(_make_transpose, [2, 48, 7, 9], [0, 2, 3, 1]),
    "transpose-nhwc2nchw-1x32x32x64xbf16": partial(_make_transpose, [1, 32, 32, 64], [0, 3, 1, 2]),
    "transpose-nhwc2nchw-2x7x9x48xbf16": partial(_make_transpose, [2, 7, 9, 48], [0, 3, 1, 2]),
}
