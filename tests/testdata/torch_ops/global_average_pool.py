"""Global average pool operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import BF16, I8, OnnxGraph

KIND = "onnx"


def make_GlobalAveragePool():
    g = OnnxGraph(opset=18)
    x = g.input("x", BF16, [1, 2560, 8, 8])
    return g.export(g.op("GlobalAveragePool", x, dtype=BF16, shape=[1, 2560, 1, 1]))


def make_quant_globalAvgPool():
    g = OnnxGraph(opset=18)
    x = g.input("x", I8, [1, 512, 7, 7])
    pooled = g.op("GlobalAveragePool", g.dequantize(x, 0.007843106053769588, 0))
    return g.export(g.quantize(pooled, 0.002344646258279681, -5, dtype=I8, shape=[1, 512, 1, 1]))


CASES = {
    "GlobalAveragePool": make_GlobalAveragePool,
    "quant-globalAvgPool": make_quant_globalAvgPool,
}

CONFIG = {
    "GlobalAveragePool": {"reference": "numpy_global_average_pool_reference_results"},
}
