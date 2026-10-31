"""Matmul operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import LazyValues

KIND = "onnx"


def _make_block_dequant_matmul(
    weight_shape, block_size, *, zero_point_range=None, zero_point=0, scale=1.0, opset=21
) -> onnx.ModelProto:
    k, n = weight_shape
    scale_shape = [(k + block_size - 1) // block_size, n]
    input_0 = gs.Variable("input_0", dtype=np.int8, shape=weight_shape)
    input_1 = gs.Variable("input_1", dtype=ml_dtypes.bfloat16, shape=[1, 1, k])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 1, n])
    scale = gs.Constant("scale", np.full(scale_shape, scale, dtype=ml_dtypes.bfloat16))
    if zero_point_range is not None:
        values = np.random.default_rng(0).integers(*zero_point_range, scale_shape, dtype=np.int8)
    else:
        values = np.full(scale_shape, zero_point, dtype=np.int8)
    zero_point = gs.Constant("zero_point", values)
    dequantizelinear = gs.Variable("dequantizelinear")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear",
                inputs=[input_0, scale, zero_point],
                outputs=[dequantizelinear],
                attrs={"axis": 0, "block_size": block_size},
            ),
            gs.Node("MatMul", inputs=[input_1, dequantizelinear], outputs=[output]),
        ],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=opset,
    )
    return gs.export_onnx(graph, ir_version=10)


def _make_matmul(
    input_shape, weight_shape, *, weight_value=None, weight_range=None, opset=22
) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable(
        "output", dtype=ml_dtypes.bfloat16, shape=[*input_shape[:-1], weight_shape[-1]]
    )
    inputs = [x]
    if weight_range is not None:
        values = (
            np.random.default_rng(0).uniform(*weight_range, weight_shape).astype(ml_dtypes.bfloat16)
        )
        weights = gs.Constant("weights", values)
    elif weight_value is not None:
        weights = gs.Constant(
            "weights", np.full(weight_shape, weight_value, dtype=ml_dtypes.bfloat16)
        )
    else:
        weights = gs.Variable("weights", dtype=ml_dtypes.bfloat16, shape=weight_shape)
        inputs.append(weights)
    graph = gs.Graph(
        nodes=[gs.Node("MatMul", inputs=[x, weights], outputs=[output])],
        inputs=inputs,
        outputs=[output],
        opset=opset,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_matmul_wi8_dequant_loop_reorder_bf16() -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=ml_dtypes.bfloat16, shape=[300, 800])
    input_1 = gs.Variable("input_1", dtype=np.int8, shape=[800, 3200])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[300, 3200])
    scale = gs.Constant("scale", np.full([], 0.015625, dtype=ml_dtypes.bfloat16))
    zero_point = gs.Constant("zero_point", np.full([], 0, dtype=np.int8))
    dequantizelinear = gs.Variable("dequantizelinear")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear", inputs=[input_1, scale, zero_point], outputs=[dequantizelinear]
            ),
            gs.Node("MatMul", inputs=[input_0, dequantizelinear], outputs=[output]),
        ],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_matmul_dql_q8_0() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 1, 640])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 1, 256])
    quantized = gs.Constant("quantized", rng.integers(-128, 128, [640, 256], dtype=np.int8))
    scale = gs.Constant(
        "scale",
        rng.uniform(0.00019168853759765625, 0.0023651123046875, [20, 256]).astype(
            ml_dtypes.bfloat16
        ),
    )
    zero_point = gs.Constant("zero_point", rng.integers(-88, 77, [20, 256], dtype=np.int8))
    dequantizelinear = gs.Variable("dequantizelinear")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized, scale, zero_point],
                outputs=[dequantizelinear],
                attrs={"axis": 0, "block_size": 32},
            ),
            gs.Node("MatMul", inputs=[x, dequantizelinear], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_tied_operands_kv_cache() -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=ml_dtypes.bfloat16, shape=[1, 1, 64])
    input_1 = gs.Variable("input_1", dtype=ml_dtypes.bfloat16, shape=[1, 4, 8, 64])
    input_2 = gs.Variable("input_2", dtype=ml_dtypes.bfloat16, shape=[1, 4, 8, 64])
    output_0 = gs.Variable("output_0", dtype=ml_dtypes.bfloat16, shape=[1, 1, 256])
    output_1 = gs.Variable("output_1", dtype=ml_dtypes.bfloat16, shape=[1, 4, 8, 64])
    output_2 = gs.Variable("output_2", dtype=ml_dtypes.bfloat16, shape=[1, 4, 8, 64])
    weights = gs.Constant("weights", np.full([64, 256], 1.0, dtype=ml_dtypes.bfloat16))
    graph = gs.Graph(
        nodes=[
            gs.Node("MatMul", inputs=[input_0, weights], outputs=[output_0]),
            gs.Node("Identity", inputs=[input_1], outputs=[output_1]),
            gs.Node("Identity", inputs=[input_2], outputs=[output_2]),
        ],
        inputs=[input_0, input_1, input_2],
        outputs=[output_0, output_1, output_2],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def _make_integer_matmul(*, weight_zero_point=0) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=np.float32, shape=[4, 1, 3])
    output = gs.Variable("output", dtype=np.float32, shape=[4, 1, 1])
    W = gs.Constant("W", rng.integers(1, 13, [4, 3, 1], dtype=np.int8))
    weight_zero_point = gs.Constant(
        "weight_zero_point", np.full([1], weight_zero_point, dtype=np.int8)
    )
    mul_constant = gs.Constant("mul_constant", np.full([1], 0.0078125, dtype=np.float32))
    dynamicquantizelinear_0 = gs.Variable("dynamicquantizelinear_0")
    dynamicquantizelinear_1 = gs.Variable("dynamicquantizelinear_1")
    dynamicquantizelinear_2 = gs.Variable("dynamicquantizelinear_2")
    matmulinteger = gs.Variable("matmulinteger")
    cast = gs.Variable("cast")
    mul = gs.Variable("mul")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DynamicQuantizeLinear",
                inputs=[x],
                outputs=[dynamicquantizelinear_0, dynamicquantizelinear_1, dynamicquantizelinear_2],
            ),
            gs.Node(
                "MatMulInteger",
                inputs=[dynamicquantizelinear_0, W, dynamicquantizelinear_2, weight_zero_point],
                outputs=[matmulinteger],
            ),
            gs.Node("Cast", inputs=[matmulinteger], outputs=[cast], attrs={"to": 1}),
            gs.Node("Mul", inputs=[dynamicquantizelinear_1, mul_constant], outputs=[mul]),
            gs.Node("Mul", inputs=[cast, mul], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=21,
    )
    return gs.export_onnx(graph, ir_version=10)


def _make_integer_qkv(input_shape, *, fused=False) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=np.float32, shape=input_shape)
    outputs = [
        gs.Variable(f"output_{i}", dtype=np.float32, shape=[*input_shape[:-1], 32])
        for i in range(3)
    ]
    quantized, scale, zero_point = [gs.Variable(f"dynamicquantizelinear_{i}") for i in range(3)]
    weight_zero_point = gs.Constant("weight_zero_point", np.full([], 0, dtype=np.int8))
    nodes = [gs.Node("DynamicQuantizeLinear", inputs=[x], outputs=[quantized, scale, zero_point])]
    if fused:
        weights = gs.Constant("weights", rng.integers(-127, 127, [16, 96], dtype=np.int8))
        matmul = gs.Variable("matmulinteger")
        nodes.append(
            gs.Node(
                "MatMulInteger",
                inputs=[quantized, weights, zero_point, weight_zero_point],
                outputs=[matmul],
            )
        )
    weight_ranges = [(-126, 127), (-127, 127), (-126, 126)]
    scales = [0.012299999594688416, 0.013530000112950802, 0.014759999699890614]
    for i, output in enumerate(outputs):
        suffix = "" if i == 0 else f"_{i + 1}"
        if fused:
            starts = gs.Constant(f"starts{suffix}", np.full([1], i * 32, dtype=np.int64))
            ends = gs.Constant(f"ends{suffix}", np.full([1], (i + 1) * 32, dtype=np.int64))
            axes = gs.Constant(f"axes{suffix}", np.full([1], 1, dtype=np.int64))
            result = gs.Variable(f"slice{suffix}")
            nodes.append(gs.Node("Slice", inputs=[matmul, starts, ends, axes], outputs=[result]))
        else:
            weights = gs.Constant(
                f"weights{suffix}", rng.integers(*weight_ranges[i], [16, 32], dtype=np.int8)
            )
            result = gs.Variable(f"matmulinteger{suffix}")
            nodes.append(
                gs.Node(
                    "MatMulInteger",
                    inputs=[quantized, weights, zero_point, weight_zero_point],
                    outputs=[result],
                )
            )
        cast = gs.Variable(f"cast{suffix}")
        mul = gs.Variable(f"mul{suffix}")
        weight_scale = gs.Constant(
            f"mul_constant{suffix}", np.full([], scales[i], dtype=np.float32)
        )
        nodes.extend(
            [
                gs.Node("Cast", inputs=[result], outputs=[cast], attrs={"to": 1}),
                gs.Node("Mul", inputs=[scale, weight_scale], outputs=[mul]),
                gs.Node("Mul", inputs=[cast, mul], outputs=[output]),
            ]
        )
    graph = gs.Graph(nodes=nodes, inputs=[x], outputs=outputs, opset=21)
    return gs.export_onnx(graph, ir_version=10)


def make_matmulinteger_u8_weight_full_range_perchannel() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=np.float32, shape=[4, 16])
    output = gs.Variable("output", dtype=np.float32, shape=[4, 32])
    W = gs.Constant(
        "W", rng.permutation(np.tile(np.arange(256, dtype=np.uint8), 2)).reshape(16, 32)
    )
    weight_zero_point = gs.Constant("weight_zero_point", np.full([], 145, dtype=np.uint8))
    mul_constant = gs.Constant(
        "mul_constant",
        rng.uniform(0.012314075604081154, 0.013529447838664055, [32]).astype(np.float32),
    )
    dynamicquantizelinear_0 = gs.Variable("dynamicquantizelinear_0")
    dynamicquantizelinear_1 = gs.Variable("dynamicquantizelinear_1")
    dynamicquantizelinear_2 = gs.Variable("dynamicquantizelinear_2")
    matmulinteger = gs.Variable("matmulinteger")
    cast = gs.Variable("cast")
    mul = gs.Variable("mul")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DynamicQuantizeLinear",
                inputs=[x],
                outputs=[dynamicquantizelinear_0, dynamicquantizelinear_1, dynamicquantizelinear_2],
            ),
            gs.Node(
                "MatMulInteger",
                inputs=[dynamicquantizelinear_0, W, dynamicquantizelinear_2, weight_zero_point],
                outputs=[matmulinteger],
            ),
            gs.Node("Cast", inputs=[matmulinteger], outputs=[cast], attrs={"to": 1}),
            gs.Node("Mul", inputs=[dynamicquantizelinear_1, mul_constant], outputs=[mul]),
            gs.Node("Mul", inputs=[cast, mul], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=21,
    )
    return gs.export_onnx(graph, ir_version=10)


def _make_int4_matmul(k, scale_range, zero_point_range, *, n=256, block_size=32) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 1, k])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 1, n])
    scale_shape = [(k + block_size - 1) // block_size, n]
    # GraphSurgeon 0.6.1 does not pack INT4 arrays; ONNX handles their storage.
    quantized_values = rng.integers(-8, 8, [k, n], dtype=np.int8).astype(ml_dtypes.int4)
    quantized = gs.Constant("quantized", LazyValues(onnx.numpy_helper.from_array(quantized_values)))
    scale = gs.Constant("scale", rng.uniform(*scale_range, scale_shape).astype(ml_dtypes.bfloat16))
    zero_point_values = rng.integers(*zero_point_range, scale_shape, dtype=np.int8).astype(ml_dtypes.int4)
    zero_point = gs.Constant("zero_point", LazyValues(onnx.numpy_helper.from_array(zero_point_values)))
    dequantizelinear = gs.Variable("dequantizelinear")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized, scale, zero_point],
                outputs=[dequantizelinear],
                attrs={"axis": 0, "block_size": block_size},
            ),
            gs.Node("MatMul", inputs=[x, dequantizelinear], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "gemm-fc-zero-bias-narrow-n-bf16": partial(
        _make_matmul, [2070, 1152], [1152, 8], weight_value=0.010009765625
    ),
    "batched_const_matmul_broadcast-bf16": partial(
        _make_matmul, [2, 1, 16], [16, 8], weight_range=(-2.28125, 2.265625)
    ),
    "dequant_matmul_zp_n6": partial(
        _make_block_dequant_matmul, [32, 6], 32, zero_point_range=(16, 90)
    ),
    "dequant_matmul_zp_nested_tile": partial(
        _make_block_dequant_matmul, [2048, 128], 1024, zero_point_range=(-100, 101)
    ),
    "dequant_matmul_zp_wide_n": partial(
        _make_block_dequant_matmul, [512, 1024], 256, zero_point_range=(-100, 101)
    ),
    "matmul-1x128x3072x768-bf16": partial(
        _make_matmul, [1, 128, 3072], [3072, 768], weight_value=0.0625, opset=18
    ),
    "matmul-1x128x768x3072-bf16": partial(
        _make_matmul, [1, 128, 768], [768, 3072], weight_value=0.0625, opset=18
    ),
    "matmul-loop-reorder-300x800x3200-bf16": partial(
        _make_matmul, [300, 800], [800, 3200], weight_value=0.010009765625
    ),
    "matmul-tile-no-vector-snap-bf16": partial(
        _make_matmul, [64, 1600], [1600, 126], weight_value=0.010009765625
    ),
    "matmul-wi8-dequant-loop-reorder-bf16": make_matmul_wi8_dequant_loop_reorder_bf16,
    "matmul_dql_q8_0": make_matmul_dql_q8_0,
    "matmul_dql_q8_0_odd_tile": partial(
        _make_block_dequant_matmul, [640, 205], 32, zero_point_range=(-84, 77), opset=22
    ),
    "matmul_dql_q8_0_zp_splat_tiled": partial(
        _make_block_dequant_matmul, [256, 1024], 32, zero_point=127, scale=0.03125
    ),
    "matmul_dql_q8_0_zp_zero_tiled": partial(
        _make_block_dequant_matmul, [256, 1024], 32, scale=0.03125
    ),
    "tied_operands_kv_cache": make_tied_operands_kv_cache,
    "matmulinteger_depthwise_conv": partial(_make_integer_matmul),
    "matmulinteger_depthwise_conv_wzp": partial(_make_integer_matmul, weight_zero_point=-128),
    "matmulinteger_fused_qkv": partial(_make_integer_qkv, [1, 16], fused=True),
    "matmulinteger_qkv_rank3": partial(_make_integer_qkv, [1, 4, 16]),
    "matmulinteger_u8_weight_full_range_perchannel": make_matmulinteger_u8_weight_full_range_perchannel,
    "matmul-rank2-bf16-f32acc": partial(_make_matmul, [4, 8], [8, 16], opset=19),
    "matmul-rank4x2-bf16-f32acc": partial(_make_matmul, [1, 2, 4, 8], [8, 16], opset=19),
    'matmul_dql_128x256_i4_zp': partial(_make_int4_matmul, 128, (0.0101318359375, 0.10009765625), (-3, 4)),
    'matmul_dql_q4_0': partial(_make_int4_matmul, 640, (0.0050048828125, 0.02001953125), (-8, 8)),
}
