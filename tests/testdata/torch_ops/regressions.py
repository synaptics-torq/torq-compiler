"""Small regression graphs preserved from compiler test inputs."""

from torq.testing.onnx_builder import BF16, F32, I64, OnnxGraph

KIND = "onnx"


def make_gelu():
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, [1, 196, 24])

    def op(kind, *inputs):
        return g.op(kind, *inputs, dtype=BF16, shape=x.shape)

    def const(value):
        return g.const(value, BF16)

    cube = op("Pow", x, const(3.0))
    scaled = op("Mul", cube, const(0.04467770))
    tanh = op("Tanh", op("Mul", op("Add", x, scaled), const(0.796875)))
    return g.export(op("Mul", op("Mul", x, const(0.5)), op("Add", tanh, const(1.0))), ir_version=8)


def make_reshape():
    g = OnnxGraph(opset=19)
    x = g.input("x", BF16, [1, 256, 20, 20])
    shape = [1, 256, 400]
    return g.export(g.op("Reshape", x, g.const(shape, I64), dtype=BF16, shape=shape))


def make_reducemax():
    # #2632: tiled partial maxima must stay on the NPU with host/CSS disabled.
    g = OnnxGraph(opset=22)
    x = g.input("x", F32, [4, 162, 1, 512])
    y = g.op("ReduceMax", x, g.const([3], I64), keepdims=1, noop_with_empty_axes=0, dtype=F32,
             shape=[4, 162, 1, 1])
    return g.export(y)


CASES = {
    "gelu-tanh-decomposed-bf16-1x196x24": make_gelu,
    "reshape-only-bf16": make_reshape,
    "reducemax-tiled-large-outer-f32": make_reducemax,
}

CONFIG = {
    # GELU can produce tiny values in the far negative tail, where the true result is already
    # effectively zero.
    "gelu-tanh-decomposed-bf16-1x196x24": {
        "comparison": {"epsilon": 2e-5},
        "reference": "numpy_gelu_reference_results",
    },
}
