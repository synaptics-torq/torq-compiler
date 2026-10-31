"""Average pool operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def make_localavgpool():
    g = OnnxGraph(opset=18)
    shape = [1, 192, 35, 35]
    x = g.input("x", BF16, shape)
    y = g.op("AveragePool", x, ceil_mode=0, count_include_pad=1, kernel_shape=[3, 3],
             pads=[1, 1, 1, 1], strides=[1, 1], dtype=BF16, shape=shape)
    return g.export(y)


CASES = {
    "localavgpool": make_localavgpool,
}

CONFIG = {
    "localavgpool": {"comparison": {"fp_avg_tol": 0.2, "fp_max_tol": 1.004}},
}
