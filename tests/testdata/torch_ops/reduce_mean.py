"""Reduce Mean operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_reduce_mean(
    input_shape, output_shape, axes, *, reshape=False, **attrs
) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=output_shape)
    axes = gs.Constant("reducemean_constant", np.array(axes, dtype=np.int64))
    reduced = gs.Variable("reducemean") if reshape else output
    nodes = [gs.Node("ReduceMean", inputs=[x, axes], outputs=[reduced], attrs=attrs)]
    if reshape:
        shape = gs.Constant("shape", np.array(output_shape, dtype=np.int64))
        nodes.append(
            gs.Node("Reshape", inputs=[reduced, shape], outputs=[output], attrs={"allowzero": 1})
        )
    graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[output], opset=22)
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "reducemean": partial(
        _make_reduce_mean,
        [1, 1280, 7, 7],
        [1, 1280, 1, 1],
        [-1, -2],
        keepdims=1,
        noop_with_empty_axes=0,
    ),
    "reducemean-reshape": partial(
        _make_reduce_mean,
        [1, 1280, 7, 7],
        [1, 1280],
        [-1, -2],
        reshape=True,
        keepdims=1,
        noop_with_empty_axes=0,
    ),
    "reducemean-tiling": partial(_make_reduce_mean, [1, 1, 359712], [1, 1, 1], [-1]),
    "reducemean_wide_kept_dims_bf16": partial(
        _make_reduce_mean, [1, 640, 3, 608], [1, 640, 608], [2], keepdims=0, noop_with_empty_axes=0
    ),
}
