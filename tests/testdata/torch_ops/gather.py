"""Gather operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_gather(
    input_shape, output_shape, *, indices=None, index_shape=None, axis=0, opset=22
) -> onnx.ModelProto:
    dynamic = indices is None
    x = gs.Variable("input_0" if dynamic else "x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=output_shape)
    if dynamic:
        indices = gs.Variable("input_1", dtype=np.int64, shape=index_shape)
    else:
        indices = gs.Constant("indices", np.array(indices, dtype=np.int64))
    graph = gs.Graph(
        nodes=[gs.Node("Gather", inputs=[x, indices], outputs=[output], attrs={"axis": axis})],
        inputs=[x, indices] if dynamic else [x],
        outputs=[output],
        opset=opset,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "gather-1d-idx-reorder-3d-bf16": partial(
        _make_gather, [16, 1, 64], [16, 1, 64], indices=list(range(15, -1, -1))
    ),
    "gather-dynamic-indices-tsuki": partial(
        _make_gather, [1, 64, 1280], [1, 64, 6401], index_shape=[6401], axis=2, opset=19
    ),
    "gather-scalar-idx-3d-select1-bf16": partial(_make_gather, [16, 1, 64], [1, 64], indices=1),
}
