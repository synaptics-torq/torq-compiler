"""Max Pool operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_max_pool(
    input_shape,
    output_shape,
    *,
    kernel_shape=(2, 2),
    strides=(2, 2),
    pads=(0, 0, 0, 0),
    explicit_defaults=True,
    ceil_mode=None,
    opset=23
) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=output_shape)
    attrs = {}
    if explicit_defaults:
        attrs.update(ceil_mode=0, dilations=[1, 1])
    if ceil_mode is not None:
        attrs["ceil_mode"] = ceil_mode
    attrs["kernel_shape"] = list(kernel_shape)
    if explicit_defaults or ceil_mode is not None:
        attrs["pads"] = list(pads)
    attrs["strides"] = list(strides)
    graph = gs.Graph(
        nodes=[gs.Node("MaxPool", inputs=[x], outputs=[output], attrs=attrs)],
        inputs=[x],
        outputs=[output],
        opset=opset,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "maxpool1d_bf16_stride4_k4": partial(
        _make_max_pool, [1, 32, 256, 64], [1, 32, 64, 64], kernel_shape=(4, 1), strides=(4, 1)
    ),
    "maxpool_bf16_batch19_2x2_stride2x2_32x16x4": partial(
        _make_max_pool, [19, 32, 16, 4], [19, 32, 8, 2]
    ),
    "maxpool_bf16_batch19_2x2_stride2x2_32x32x8": partial(
        _make_max_pool, [19, 32, 32, 8], [19, 32, 16, 4]
    ),
    "maxpool_bf16_batch19_2x2_stride2x2_32x64x16": partial(
        _make_max_pool, [19, 32, 64, 16], [19, 32, 32, 8]
    ),
    "maxpool_bf16_batch19_4x1_stride4x1": partial(
        _make_max_pool, [19, 32, 256, 16], [19, 32, 64, 16], kernel_shape=(4, 1), strides=(4, 1)
    ),
    "maxpool_bf16_batch1_1x2_stride1x2_64x1x2": partial(
        _make_max_pool,
        [1, 64, 1, 2],
        [1, 64, 1, 1],
        kernel_shape=(1, 2),
        strides=(1, 2),
        explicit_defaults=False,
        opset=22,
    ),
    "maxpool_bf16_batch1_2x2_stride2x2_16x33x20": partial(
        _make_max_pool, [1, 16, 33, 20], [1, 16, 16, 10], explicit_defaults=False, opset=22
    ),
    "maxpool_bf16_batch1_2x2_stride2x2_16x66x41": partial(
        _make_max_pool, [1, 16, 66, 41], [1, 16, 33, 20], explicit_defaults=False, opset=22
    ),
    "maxpool_bf16_batch1_2x2_stride2x2_32x16x4": partial(
        _make_max_pool, [1, 32, 16, 4], [1, 32, 8, 2]
    ),
    "maxpool_bf16_batch1_2x2_stride2x2_32x32x8": partial(
        _make_max_pool, [1, 32, 32, 8], [1, 32, 16, 4]
    ),
    "maxpool_bf16_batch1_2x2_stride2x2_32x64x16": partial(
        _make_max_pool, [1, 32, 64, 16], [1, 32, 32, 8]
    ),
    "maxpool_bf16_batch1_2x2_stride2x2_32x8x64": partial(
        _make_max_pool, [1, 32, 8, 64], [1, 32, 4, 32]
    ),
    "maxpool_bf16_batch1_3x3_stride1x1_16x4": partial(
        _make_max_pool,
        [1, 16, 4, 32],
        [1, 16, 4, 32],
        kernel_shape=(3, 3),
        strides=(1, 1),
        pads=(1, 1, 1, 1),
    ),
    "maxpool_bf16_batch1_3x3_stride2x2_pad1_3x241x320": partial(
        _make_max_pool,
        [1, 3, 241, 320],
        [1, 3, 121, 160],
        kernel_shape=(3, 3),
        pads=(1, 1, 1, 1),
        explicit_defaults=False,
        opset=22,
        ceil_mode=0,
    ),
}
