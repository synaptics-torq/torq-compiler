"""SplitConstantsPass must fold each static subview of a constant into a constant holding exactly
the selected elements, for every kind of subview (contiguous, strided, rank-reduced, scalar) and
element type (both the raw-copy path and the per-element fallback)."""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

# (name, source shape, offsets, sizes, strides, dims dropped from the result)
SLICES = [
    ("rows", (4, 8), (1, 0), (2, 8), (1, 1), ()),
    ("inner_part", (4, 8), (0, 2), (4, 4), (1, 1), ()),
    ("outer_strided", (6, 8), (1, 0), (3, 8), (2, 1), ()),
    ("inner_strided", (4, 8), (0, 1), (4, 4), (1, 2), ()),
    ("unit_inner_strided", (4, 8), (0, 3), (4, 1), (1, 2), ()),
    ("unit_dims_3d", (2, 3, 4), (1, 1, 1), (1, 2, 3), (1, 1, 1), ()),
    ("rank_reduced_row", (4, 8), (2, 0), (1, 8), (1, 1), (0,)),
    ("rank_reduced_middle", (3, 4, 5), (0, 1, 0), (3, 1, 5), (1, 1, 1), (1,)),
    ("rank_reduced_part", (3, 4, 5), (1, 0, 2), (1, 4, 2), (1, 1, 1), (0,)),
    ("scalar", (4, 8), (3, 5), (1, 1), (1, 1), (0, 1)),
]

# MLIR element type, numpy type of the expected values
TYPES = [
    ("i4", np.int8),
    ("i8", np.int8),
    ("i16", np.int16),
    ("i32", np.int32),
    ("i64", np.int64),
    ("bf16", np.float32),
    ("f32", np.float32),
    ("i1", np.bool_),
]


def source_values(shape, element_type, np_type, splat):
    n = int(np.prod(shape))
    if np_type == np.bool_:
        values = np.arange(n) % 3 == 0
    elif np.issubdtype(np_type, np.floating):
        values = np.arange(n, dtype=np_type) * 0.5 - 3  # exact in bf16
    else:
        bits = min(int(element_type[1:]), 16)
        values = (np.arange(n) % (1 << bits) - (1 << (bits - 1))).astype(np_type)
    if splat:
        values = np.full(n, values[1])
    return values.reshape(shape)


def mlir_dense(values, element_type):
    def fmt(v):
        if element_type == "i1":
            return "true" if v else "false"
        return repr(float(v)) if element_type in ("bf16", "f32") else str(int(v))

    def nest(a):
        return fmt(a) if a.ndim == 0 else "[" + ", ".join(nest(x) for x in a) + "]"

    return nest(values)


def memref_type(shape, element_type, layout=""):
    dims = "".join(f"{d}x" for d in shape)
    return f"memref<{dims}{element_type}{layout}>"


def make_module(shape, offsets, sizes, strides, dropped, element_type, values):
    src_strides = [int(np.prod(shape[i + 1:])) for i in range(len(shape))]
    offset = sum(o * s for o, s in zip(offsets, src_strides))
    kept = [i for i in range(len(shape)) if i not in dropped]
    res_shape = [sizes[i] for i in kept]
    res_strides = [src_strides[i] * strides[i] for i in kept]
    layout = f", strided<[{', '.join(map(str, res_strides))}], offset: {offset}>"
    src_ty = memref_type(shape, element_type)
    res_ty = memref_type(res_shape, element_type, layout)
    out_ty = memref_type(res_shape, element_type)
    tensor_ty = "tensor<" + "".join(f"{d}x" for d in shape) + element_type + ">"
    dense = mlir_dense(values, element_type)
    if values.size > 1 and np.all(values == values.flat[0]):
        dense = mlir_dense(values.flat[0], element_type)
    return f"""
func.func @slice(%out: {out_ty}) {{
  %c = "torq_hl.const"() <{{value = dense<{dense}> : {tensor_ty}}}> : () -> {src_ty}
  %s = memref.subview %c[{', '.join(map(str, offsets))}] [{', '.join(map(str, sizes))}] [{', '.join(map(str, strides))}] : {src_ty} to {res_ty}
  memref.copy %s, %out : {res_ty} to {out_ty}
  return
}}
"""


def folded_values(ir, element_type):
    consts = re.findall(r'"?torq_hl\.const"?.*?dense<(.*?)> : tensor<([^>]*)>', ir)
    assert len(consts) == 1, f"expected one constant after the pass:\n{ir}"
    body, ty = consts[0]
    shape = [int(d) for d in ty.split("x")[:-1]]
    if element_type == "i1":
        flat = [t == "true" for t in re.findall(r"true|false", body)]
    else:
        flat = [float(t) for t in re.findall(r"-?\d+(?:\.\d*)?(?:e[+-]?\d+)?", body)]
    if len(flat) == 1:  # splat
        flat = flat * int(np.prod(shape))
    return np.array(flat).reshape(shape)


@pytest.mark.ci
@pytest.mark.parametrize("splat", [False, True], ids=["dense", "splat"])
@pytest.mark.parametrize("element_type,np_type", TYPES, ids=[t[0] for t in TYPES])
@pytest.mark.parametrize("name,shape,offsets,sizes,strides,dropped", SLICES, ids=[s[0] for s in SLICES])
def test_split_constant_subview(
    torq_compiler, name, shape, offsets, sizes, strides, dropped, element_type, np_type, splat
):
    values = source_values(shape, element_type, np_type, splat)
    source = make_module(shape, offsets, sizes, strides, dropped, element_type, values)
    optimizer = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [
            str(optimizer),
            "--torq-enable-split-constants-optimization",
            "--mlir-print-elementsattrs-with-hex-if-larger=-1",
            "--pass-pipeline=builtin.module(func.func(torq-split-constants))",
        ],
        input=source, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "memref.subview" not in proc.stdout, proc.stdout

    index = tuple(
        o if d in dropped else slice(o, o + (s - 1) * st + 1, st)
        for d, (o, s, st) in enumerate(zip(offsets, sizes, strides))
    )
    expected = values[index]
    got = folded_values(proc.stdout, element_type)
    assert got.shape == expected.shape, proc.stdout
    np.testing.assert_array_equal(got, expected.astype(got.dtype))
