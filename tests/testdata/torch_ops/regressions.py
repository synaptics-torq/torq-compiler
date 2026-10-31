"""Small regression graphs preserved from compiler test inputs."""

import ml_dtypes
import numpy as np
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_gelu():
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 196, 24])
    nodes = []

    def op(kind, *inputs):
        out = gs.Variable(f"v{len(nodes)}", dtype=x.dtype, shape=x.shape)
        nodes.append(gs.Node(kind, inputs=list(inputs), outputs=[out]))
        return out

    def const(name, value):
        return gs.Constant(name, np.array(value, dtype=x.dtype))

    half, one = const("half", 0.5), const("one", 1.0)
    cube = op("Pow", x, const("three", 3.0))
    scaled = op("Mul", cube, const("coeff", 0.04467770))
    tanh = op("Tanh", op("Mul", op("Add", x, scaled), const("sqrt2pi", 0.796875)))
    out = op("Mul", op("Mul", x, half), op("Add", tanh, one))
    return gs.export_onnx(gs.Graph(nodes=nodes, inputs=[x], outputs=[out], opset=22), ir_version=8)


def make_reshape():
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 256, 20, 20])
    out = gs.Variable("out", dtype=x.dtype, shape=[1, 256, 400])
    shape = gs.Constant("shape", np.array(out.shape, dtype=np.int64))
    return gs.export_onnx(gs.Graph(
        nodes=[gs.Node("Reshape", inputs=[x, shape], outputs=[out])],
        inputs=[x], outputs=[out], opset=19), ir_version=10)


def make_reducemax():
    # #2632: tiled partial maxima must stay on the NPU with host/CSS disabled.
    x = gs.Variable("x", dtype=np.float32, shape=[4, 162, 1, 512])
    out = gs.Variable("out", dtype=x.dtype, shape=[4, 162, 1, 1])
    axis = gs.Constant("axis", np.array([3], dtype=np.int64))
    return gs.export_onnx(gs.Graph(
        nodes=[gs.Node("ReduceMax", inputs=[x, axis], outputs=[out],
                       attrs={"keepdims": 1, "noop_with_empty_axes": 0})],
        inputs=[x], outputs=[out], opset=22), ir_version=10)


CASES = {
    "gelu-tanh-decomposed-bf16-1x196x24": make_gelu,
    "reshape-only-bf16": make_reshape,
    "reducemax-tiled-large-outer-f32": make_reducemax,
}
