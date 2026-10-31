"""Pow operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_pow(input_shape, *, exponent=2.0) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=input_shape)
    exponent = gs.Constant("exponent", np.full([], exponent, dtype=ml_dtypes.bfloat16))
    graph = gs.Graph(
        nodes=[gs.Node("Pow", inputs=[x, exponent], outputs=[output])],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "pow-half-scalar-bf16": partial(_make_pow, [1, 1], exponent=0.5),
    "pow2-1x207x288xbf16": partial(_make_pow, [1, 207, 288]),
    "pow2-scalar-bf16": partial(_make_pow, [1, 1, 1]),
}
