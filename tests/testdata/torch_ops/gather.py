"""Gather operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import BF16, I64, OnnxGraph

KIND = "onnx"


def _make_gather(
    input_shape, output_shape, *, indices=None, index_shape=None, axis=0, opset=22
):
    g = OnnxGraph(opset)
    if indices is None:
        x = g.input("input_0", BF16, input_shape)
        indices = g.input("input_1", I64, index_shape)
    else:
        x = g.input("x", BF16, input_shape)
        indices = g.const(indices, I64)
    return g.export(g.op("Gather", x, indices, axis=axis, dtype=BF16, shape=output_shape))


CASES = {
    "gather-1d-idx-reorder-3d-bf16": partial(
        _make_gather, [16, 1, 64], [16, 1, 64], indices=list(range(15, -1, -1))
    ),
    "gather-dynamic-indices-tsuki": partial(
        _make_gather, [1, 64, 1280], [1, 64, 6401], index_shape=[6401], axis=2, opset=19
    ),
    "gather-scalar-idx-3d-select1-bf16": partial(_make_gather, [16, 1, 64], [1, 64], indices=1),
}

CONFIG = {
    # Must lower the value copy to torq_hl.gather on the NSS. CSS stays enabled: the ONNX Gather
    # index math (negative-index normalization on i64 and the i64->i32 demote) has no NSS data
    # path and must run on CSS.
    "gather-dynamic-indices-tsuki": {"compiler_options": ["--torq-disable-host"]},
}
