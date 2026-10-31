"""Reduce Mean operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import BF16, I64, OnnxGraph, typed

KIND = "onnx"


def _make_reduce_mean(input_shape, output_shape, axes, *, reshape=False, **attrs):
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, input_shape)
    y = g.op("ReduceMean", x, g.const(axes, I64), **attrs)
    if reshape:
        y = g.op("Reshape", y, g.const(output_shape, I64), allowzero=1)
    return g.export(typed(y, BF16, output_shape))


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

# The NPU accumulates reductions in fp32, while the llvm-cpu reference accumulates in bf16. For
# large reductions, or small ones over large inputs that cancel, the reference is the inaccurate
# side, so these cases use an fp32-accumulated numpy reference.
CONFIG = {
    "reducemean": {"comparison": {"fp_max_tol": 0.31, "fp_avg_tol": 0.03}},
    "reducemean-reshape": {"comparison": {"fp_max_tol": 0.31, "fp_avg_tol": 0.03}},
    "reducemean-tiling": {
        "comparison": {"fp_max_tol": 0.091},
        "reference": "numpy_reducemean_reference_results",
    },
    # LRAM-limit regression pinned to the no-slicing path used for real models (the sliced path
    # splits the reduction across slices and has an unrelated numeric mismatch at this width).
    "reducemean_wide_kept_dims_bf16": {
        "compiler_options": [
            "--torq-disable-slicing", "--torq-disable-host", "--torq-disable-css",
        ],
        "comparison": {"fp_max_tol": 0.06, "fp_avg_tol": 0.02},
        "reference": "numpy_reducemean_reference_results",
    },
}
