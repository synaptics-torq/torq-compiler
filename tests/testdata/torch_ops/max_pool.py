"""Max Pool operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import BF16, OnnxGraph

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
):
    attrs = {}
    if explicit_defaults:
        attrs.update(ceil_mode=0, dilations=[1, 1])
    if ceil_mode is not None:
        attrs["ceil_mode"] = ceil_mode
    attrs["kernel_shape"] = list(kernel_shape)
    if explicit_defaults or ceil_mode is not None:
        attrs["pads"] = list(pads)
    attrs["strides"] = list(strides)
    g = OnnxGraph(opset)
    x = g.input("x", BF16, input_shape)
    return g.export(g.op("MaxPool", x, dtype=BF16, shape=output_shape, **attrs))


def _pooled_shape(hwc, kernel_shape, strides, pads):
    """The NCHW input and output shapes of a pool of the (H, W, C) frame, with TOSA pads (top,
    bottom, left, right)."""
    (h, w, c), (kh, kw), (sh, sw), (t, b, l, r) = hwc, kernel_shape, strides, pads
    return [1, c, h, w], [1, c, (h + t + b - kh) // sh + 1, (w + l + r - kw) // sw + 1]


def _make_tosa_ops_max_pool(hwc, kernel_shape, strides, pads=(0, 0, 0, 0)):
    """A max pool ported from tests/testdata/tosa_ops: hwc is the (H, W, C) frame of the NHWC MLIR
    file and pads are (top, bottom, left, right), as TOSA spells them."""
    input_shape, output_shape = _pooled_shape(hwc, kernel_shape, strides, pads)
    t, b, l, r = pads
    return _make_max_pool(input_shape, output_shape, kernel_shape=kernel_shape, strides=strides,
                          pads=(t, l, b, r))


def _make_max_pools(hwc, kernels):
    """Max pools of the same input with SAME pads and stride 1, one output per kernel (SPP)."""
    g = OnnxGraph(23)
    input_shape, _ = _pooled_shape(hwc, (1, 1), (1, 1), (0, 0, 0, 0))
    x = g.input("x", BF16, input_shape)
    return g.export(*[
        g.op("MaxPool", x, dtype=BF16, shape=input_shape, kernel_shape=[k, k], strides=[1, 1],
             pads=[k // 2] * 4)
        for k in kernels
    ])


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
    # Ported from tests/testdata/tosa_ops, with the names of the MLIR files ((H, W, C) frames)
    "maxpool_bf16_batch1_5x5_stride1x1_192x10x10": partial(
        _make_tosa_ops_max_pool, (10, 10, 192), (5, 5), (1, 1), (2, 2, 2, 2)
    ),
    "maxpool_f13_20x20x8-bf16": partial(
        _make_tosa_ops_max_pool, (20, 20, 8), (13, 13), (1, 1), (6, 6, 6, 6)
    ),
    "maxpool_k1_s2_21x21x16-bf16": partial(_make_tosa_ops_max_pool, (21, 21, 16), (1, 1), (2, 2)),
    # Swapped to the vertical kernel, whose rows have 18 elements
    "maxpool_k1x2_s1x2_18x32x8-bf16": partial(
        _make_tosa_ops_max_pool, (18, 32, 8), (1, 2), (1, 2)
    ),
    "maxpool_k1x4_s1x4_64x64x16-bf16": partial(
        _make_tosa_ops_max_pool, (64, 64, 16), (1, 4), (1, 4)
    ),
    # Vertical 1D kernels with rows of more than one ACT vector of 16 elements
    "maxpool_k2x1_s2x1_16x18x8-bf16": partial(
        _make_tosa_ops_max_pool, (16, 18, 8), (2, 1), (2, 1)
    ),
    "maxpool_k2x1_s2x1_16x24x8-bf16": partial(
        _make_tosa_ops_max_pool, (16, 24, 8), (2, 1), (2, 1)
    ),
    "maxpool_k3_s4_pad1_21x21x16-bf16": partial(
        _make_tosa_ops_max_pool, (21, 21, 16), (3, 3), (4, 4), (1, 1, 1, 1)
    ),
    "maxpool_k3x5_s2_same_11x11x8-bf16": partial(
        _make_tosa_ops_max_pool, (11, 11, 8), (3, 5), (2, 2), (1, 1, 2, 2)
    ),
    # Split in a 3x3 stride-1 max pool and two 1D stride-3 max pools
    "maxpool_k5_s3_pad1_18x18x16-bf16": partial(
        _make_tosa_ops_max_pool, (18, 18, 16), (5, 5), (3, 3), (1, 1, 1, 1)
    ),
    # Leading padding k / 2
    "maxpool_k6_s2_pad32_21x21x8-bf16": partial(
        _make_tosa_ops_max_pool, (21, 21, 8), (6, 6), (2, 2), (3, 2, 3, 2)
    ),
    "maxpool_spp_5_9_13_13x13x32-bf16": partial(_make_max_pools, (13, 13, 32), (5, 9, 13)),
}
