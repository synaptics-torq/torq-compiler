"""Cast operator fixtures built with torq.testing.onnx_builder.

The cases are ported from tests/testdata/tosa_ops, with the names of the MLIR files. Float to
integer casts stay there: TOSA rounds them to nearest even and saturates, while an ONNX Cast
truncates, and neither a plain Cast nor Round, Clip then Cast compiles on the slices.
"""

from functools import partial

import numpy as np
import onnx

from torq.testing.onnx_builder import BF16, F32, I8, I32, OnnxGraph

KIND = "onnx"

I16 = np.int16


def _make_cast(shape, src, dst):
    g = OnnxGraph(opset=22)
    x = g.input("x", src, shape)
    to = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dst))
    return g.export(g.op("Cast", x, to=to, dtype=dst, shape=shape))


CASES = {
    "cast-bf16-to-f32": partial(_make_cast, [16, 6], BF16, F32),
    "cast-f32-to-bf16": partial(_make_cast, [160], F32, BF16),
    "cast-i16-to-bf16": partial(_make_cast, [16, 8, 3], I16, BF16),
    "cast-i16-to-f32": partial(_make_cast, [16, 35], I16, F32),
    "cast-i32-to-bf16": partial(_make_cast, [16], I32, BF16),
    "cast-i32-to-f32": partial(_make_cast, [16, 7], I32, F32),
    "cast-i8-to-bf16": partial(_make_cast, [1, 256, 16], I8, BF16),
    "cast-i8-to-f32": partial(_make_cast, [16000], I8, F32),
}
