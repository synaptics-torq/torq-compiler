"""Dynamic quantize operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_dynamic_quantize(input_shape) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.float32, shape=input_shape)
    output = gs.Variable("output", dtype=np.uint8, shape=input_shape)
    dynamicquantizelinear_1 = gs.Variable("dynamicquantizelinear_1")
    dynamicquantizelinear_2 = gs.Variable("dynamicquantizelinear_2")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DynamicQuantizeLinear",
                inputs=[x],
                outputs=[output, dynamicquantizelinear_1, dynamicquantizelinear_2],
            )
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "dynamicquantize": partial(_make_dynamic_quantize, [1, 1, 288]),
    "dynamicquantize_large": partial(_make_dynamic_quantize, [1, 1, 131072]),
}
