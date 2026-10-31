"""Is nan operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def make_isnan_check_1x4x1x256():
    g = OnnxGraph(opset=20)
    x = g.input("x", BF16, [1, 4, 1, 256])
    softmax = g.op("Softmax", x)
    isnan = g.op("IsNaN", softmax)
    zero = g.const([0.0], BF16)
    return g.export(g.op("Where", isnan, zero, softmax, dtype=BF16, shape=[1, 4, 1, 256]))


CASES = {
    "isnan_check-1x4x1x256": make_isnan_check_1x4x1x256,
}

CONFIG = {
    "isnan_check-1x4x1x256": {
        "compiler_options": [
            "--torq-disable-host", "--torq-disable-css", "--torq-isnan-mode=remove",
        ],
    },
}
