"""Softmax operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_softmax(shape, **attrs) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=shape)
    graph = gs.Graph(
        nodes=[gs.Node("Softmax", inputs=[x], outputs=[output], attrs=attrs)],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "softmax-1x8x1x207xbf16": partial(_make_softmax, [1, 8, 1, 207], axis=-1),
    "softmax-1x2xbf16": partial(_make_softmax, [1, 2]),
}
