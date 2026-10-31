"""Matmul operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, F32, I4, I8, I64, OnnxGraph

KIND = "onnx"


def _make_block_dequant_matmul(
    weight_shape, block_size, *, zero_point_range=None, zero_point=0, scale=1.0, opset=21
):
    k, n = weight_shape
    scale_shape = [(k + block_size - 1) // block_size, n]
    g = OnnxGraph(opset)
    quantized = g.input("input_0", I8, weight_shape)
    x = g.input("input_1", BF16, [1, 1, k])
    if zero_point_range is not None:
        rng = np.random.default_rng(0)
        zero_points = rng.integers(*zero_point_range, scale_shape, dtype=np.int8)
    else:
        zero_points = np.full(scale_shape, zero_point)
    weights = g.dequantize(quantized, g.const(np.full(scale_shape, scale), BF16),
                           g.const(zero_points, I8), axis=0, block_size=block_size)
    return g.export(g.op("MatMul", x, weights, dtype=BF16, shape=[1, 1, n]))


def _make_matmul(input_shape, weight_shape, *, weight_value=None, weight_range=None, opset=22):
    g = OnnxGraph(opset)
    x = g.input("x", BF16, input_shape)
    if weight_range is not None:
        weights = g.const(np.random.default_rng(0).uniform(*weight_range, weight_shape), BF16)
    elif weight_value is not None:
        weights = g.const(np.full(weight_shape, weight_value), BF16)
    else:
        weights = g.input("weights", BF16, weight_shape)
    shape = [*input_shape[:-1], weight_shape[-1]]
    return g.export(g.op("MatMul", x, weights, dtype=BF16, shape=shape))


def make_matmul_wi8_dequant_loop_reorder_bf16():
    g = OnnxGraph(opset=22)
    x = g.input("input_0", BF16, [300, 800])
    quantized = g.input("input_1", I8, [800, 3200])
    weights = g.dequantize(quantized, g.const(0.015625, BF16), 0)
    return g.export(g.op("MatMul", x, weights, dtype=BF16, shape=[300, 3200]))


def make_matmul_dql_q8_0():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, [1, 1, 640])
    quantized = g.const(rng.integers(-128, 128, [640, 256], dtype=np.int8), I8)
    scale = g.const(rng.uniform(0.00019168853759765625, 0.0023651123046875, [20, 256]), BF16)
    zero_point = g.const(rng.integers(-88, 77, [20, 256], dtype=np.int8), I8)
    weights = g.dequantize(quantized, scale, zero_point, axis=0, block_size=32)
    return g.export(g.op("MatMul", x, weights, dtype=BF16, shape=[1, 1, 256]))


def make_tied_operands_kv_cache():
    g = OnnxGraph(opset=22)
    x = g.input("input_0", BF16, [1, 1, 64])
    cache = [g.input(f"input_{i}", BF16, [1, 4, 8, 64]) for i in (1, 2)]
    y = g.op("MatMul", x, g.const(np.ones([64, 256]), BF16), dtype=BF16, shape=[1, 1, 256])
    return g.export(y, *(g.op("Identity", c, dtype=BF16, shape=[1, 4, 8, 64]) for c in cache))


def _rescale(g, product, scale, weight_scale, shape):
    """The f32 result of a MatMulInteger product: product * (input scale * weight scale)."""
    return g.op("Mul", g.op("Cast", product, to=1), g.op("Mul", scale, weight_scale), dtype=F32,
                shape=shape)


def _make_integer_matmul(*, weight_zero_point=0):
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=21)
    x = g.input("x", F32, [4, 1, 3])
    quantized, scale, zero_point = g.op("DynamicQuantizeLinear", x, outputs=3)
    weights = g.const(rng.integers(1, 13, [4, 3, 1], dtype=np.int8), I8)
    product = g.op("MatMulInteger", quantized, weights, zero_point,
                   g.const([weight_zero_point], I8))
    return g.export(_rescale(g, product, scale, g.const([0.0078125], F32), [4, 1, 1]))


def _make_integer_qkv(input_shape, *, fused=False):
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=21)
    x = g.input("x", F32, input_shape)
    quantized, scale, zero_point = g.op("DynamicQuantizeLinear", x, outputs=3)
    weight_zero_point = g.const(0, I8)
    if fused:
        weights = g.const(rng.integers(-127, 127, [16, 96], dtype=np.int8), I8)
        qkv = g.op("MatMulInteger", quantized, weights, zero_point, weight_zero_point)
    weight_ranges = [(-126, 127), (-127, 127), (-126, 126)]
    scales = [0.012299999594688416, 0.013530000112950802, 0.014759999699890614]
    outputs = []
    for i in range(3):
        if fused:
            product = g.op("Slice", qkv, g.const([i * 32], I64), g.const([(i + 1) * 32], I64),
                           g.const([1], I64))
        else:
            weights = g.const(rng.integers(*weight_ranges[i], [16, 32], dtype=np.int8), I8)
            product = g.op("MatMulInteger", quantized, weights, zero_point, weight_zero_point)
        outputs.append(
            _rescale(g, product, scale, g.const(scales[i], F32), [*input_shape[:-1], 32])
        )
    return g.export(*outputs)


def make_matmulinteger_u8_weight_full_range_perchannel():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=21)
    x = g.input("x", F32, [4, 16])
    quantized, scale, zero_point = g.op("DynamicQuantizeLinear", x, outputs=3)
    weights = rng.permutation(np.tile(np.arange(256, dtype=np.uint8), 2)).reshape(16, 32)
    product = g.op("MatMulInteger", quantized, g.const(weights, np.uint8), zero_point,
                   g.const(145, np.uint8))
    weight_scale = g.const(rng.uniform(0.012314075604081154, 0.013529447838664055, [32]), F32)
    return g.export(_rescale(g, product, scale, weight_scale, [4, 32]))


def _make_int4_matmul(k, scale_range, zero_point_range, *, n=256, block_size=32):
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, [1, 1, k])
    scale_shape = [(k + block_size - 1) // block_size, n]
    quantized = g.const(rng.integers(-8, 8, [k, n], dtype=np.int8), I4)
    scale = g.const(rng.uniform(*scale_range, scale_shape), BF16)
    zero_point = g.const(rng.integers(*zero_point_range, scale_shape, dtype=np.int8), I4)
    weights = g.dequantize(quantized, scale, zero_point, axis=0, block_size=block_size)
    return g.export(g.op("MatMul", x, weights, dtype=BF16, shape=[1, 1, n]))


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
    "matmulinteger_u8_weight_full_range_perchannel": (
        make_matmulinteger_u8_weight_full_range_perchannel
    ),
    "matmul-rank2-bf16-f32acc": partial(_make_matmul, [4, 8], [8, 16], opset=19),
    "matmul-rank4x2-bf16-f32acc": partial(_make_matmul, [1, 2, 4, 8], [8, 16], opset=19),
    "matmul_dql_128x256_i4_zp": partial(
        _make_int4_matmul, 128, (0.0101318359375, 0.10009765625), (-3, 4)
    ),
    "matmul_dql_q4_0": partial(_make_int4_matmul, 640, (0.0050048828125, 0.02001953125), (-8, 8)),
}

# matmul_dql: block quantization rounds at block boundaries, which amplifies the relative error
# of small outputs.
# matmulinteger: ONNX DynamicQuantizeLinear tensors are always f32, so the DQL -> MatMulInteger
# -> dequant chain is only reachable in a bf16 pipeline with --torq-convert-dtypes.
CONFIG = {
    "dequant_matmul_zp_n6": {"comparison": {"fp_max_tol": 0.1, "allowed_wrong": 0.1}},
    "dequant_matmul_zp_nested_tile": {"comparison": {"fp_max_tol": 0.1, "allowed_wrong": 0.1}},
    "dequant_matmul_zp_wide_n": {"comparison": {"fp_max_tol": 0.1, "allowed_wrong": 0.1}},
    # K-split repros: the NSS accumulates the chunks in fp32 and rounds once, so the reference
    # is a numpy matmul in fp32 (the llvm-cpu reference accumulates in bf16). Positive inputs
    # keep cancellation noise from dominating the comparison.
    "matmul-1x128x3072x768-bf16": {
        "input_range": [0, 2],
        "comparison": {"fp_avg_tol": 0.05, "fp_max_tol": 0.15, "allowed_wrong": 0.01},
        "reference": "numpy_matmul_const_weight_reference_results",
    },
    "matmul-1x128x768x3072-bf16": {"reference": "numpy_matmul_const_weight_reference_results"},
    "matmul_dql_128x256_i4_zp": {"comparison": {"fp_max_tol": 0.5, "allowed_wrong": 0.2}},
    "matmul_dql_q4_0": {"comparison": {"fp_max_tol": 0.5, "allowed_wrong": 0.2}},
    "matmul_dql_q8_0": {"comparison": {"fp_max_tol": 0.5, "allowed_wrong": 0.2}},
    "matmul_dql_q8_0_odd_tile": {"comparison": {"fp_max_tol": 0.5, "allowed_wrong": 0.2}},
    "matmul_dql_q8_0_zp_splat_tiled": {"comparison": {"fp_max_tol": 0.5, "allowed_wrong": 0.2}},
    "matmul_dql_q8_0_zp_zero_tiled": {"comparison": {"fp_max_tol": 0.5, "allowed_wrong": 0.2}},
    "matmulinteger_depthwise_conv": {
        "compiler_options": [
            "--torq-disable-host", "--torq-disable-css", "--torq-convert-dtypes",
            "--torq-convert-io-dtype",
        ],
        "comparison": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.015},
    },
    "matmulinteger_depthwise_conv_wzp": {
        "compiler_options": [
            "--torq-disable-host", "--torq-disable-css", "--torq-convert-dtypes",
            "--torq-convert-io-dtype",
        ],
        "comparison": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.015},
    },
    "matmulinteger_fused_qkv": {
        "compiler_options": [
            "--torq-disable-host", "--torq-disable-css", "--torq-convert-dtypes",
            "--torq-convert-io-dtype",
        ],
        "comparison": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.015},
    },
    "matmulinteger_qkv_rank3": {
        "compiler_options": [
            "--torq-disable-host", "--torq-disable-css", "--torq-convert-dtypes",
            "--torq-convert-io-dtype",
        ],
        "comparison": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.015},
    },
    "matmulinteger_u8_weight_full_range_perchannel": {
        "compiler_options": [
            "--torq-disable-host", "--torq-disable-css", "--torq-convert-dtypes",
            "--torq-convert-io-dtype",
        ],
        "comparison": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01},
    },
}
