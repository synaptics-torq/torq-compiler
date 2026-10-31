"""Sub operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_sub(input_0_shape, input_1_shape) -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=np.int32, shape=input_0_shape)
    input_1 = gs.Variable("input_1", dtype=np.int32, shape=input_1_shape)
    output = gs.Variable(
        "output", dtype=np.int32, shape=np.broadcast_shapes(input_0_shape, input_1_shape)
    )
    graph = gs.Graph(
        nodes=[gs.Node("Sub", inputs=[input_0, input_1], outputs=[output])],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=19,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "scalar_into_higher_rank_midunit": partial(_make_sub, [], [2, 1, 4]),
    "sub_scalar_tensor_nss_i32": partial(_make_sub, [], [16010]),
    "tensor_minus_scalar_rhs": partial(_make_sub, [1, 256], []),
}
