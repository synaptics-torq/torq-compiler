"""A scalar add on rows with gaps between them must use full ALU vectors and write only the
items of each row.

Each row has 197 bf16 items (6 full vectors of 32 and a partial vector of 5). A gapped buffer
has a row stride of 256 items. Before the scalar add builders were merged, a gapped input went
to a builder that used blocks of one item, because 197 is a prime number (#2722).
"""

import re
import subprocess
from pathlib import Path

import pytest

ROWS = 36
ROW_SIZE = 197
GAPPED_ROW_STRIDE = 256
BF16_BYTES = 2
# One bf16 ALU pass holds 32 items: a row is 6 full vectors and a partial vector of 5.
VECTOR_SIZE = 32
VECTORS_PER_ROW = -(-ROW_SIZE // VECTOR_SIZE)

LRAM = "#torq_hl<enc mem_space = lram>"
DENSE = f"memref<1x1x{ROWS}x{ROW_SIZE}xbf16, {LRAM}>"
GAPPED = (f"memref<1x1x{ROWS}x{ROW_SIZE}xbf16, "
          f"strided<[{ROWS * GAPPED_ROW_STRIDE}, {ROWS * GAPPED_ROW_STRIDE}, "
          f"{GAPPED_ROW_STRIDE}, 1]>, {LRAM}>")

SLICE_TO_TORQHW = (
    "builtin.module(func.func(torq_hl.program(torq-convert-slice-program-to-torqhw)))")


def scalar_add_program(input_type, output_type):
    """A slice program with one `torq_hl.add` of a bf16 scalar. The first operand is the
    output (init), the fourth operand is the input."""
    w = f"memref<2xbf16, {LRAM}>"
    sb = f"memref<1xf32, {LRAM}>"
    s = f"memref<1xbf16, {LRAM}>"
    return f"""
func.func @scalar_add() {{
  %0 = torq_hl.program "scalar_add" : !torq_hl.program<slice> {{
  ^bb0(%out: {output_type}, %w: {w}, %sb: {sb}, %in: {input_type}, %s: {s}):
    "torq_hl.add"(%out, %w, %sb, %in, %s) <{{input_zp = 0 : i32, name = "add",
        output_max = 2139095039 : i32, output_min = -8388609 : i32, output_zp = 0 : i32,
        rhs_is_scalar = true, segment_output = false, shift_factor = 0 : i32}}>
        : ({output_type}, {w}, {sb}, {input_type}, {s}) -> ()
    torq_hl.return
  }}
  return
}}
"""


def convert_to_torqhw(torq_compiler, source):
    optimizer = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [str(optimizer), f"--pass-pipeline={SLICE_TO_TORQHW}"],
        input=source, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def ndl_dims(ir, ndl):
    """The (tag, count, stride) dims of the memory NDL `ndl`, with the `#mapN` strides
    replaced by their constant values."""
    maps = dict(re.findall(r"^(#map\d*) = affine_map<\(\) -> \((-?\d+)\)>", ir, re.M))
    match = re.search(r"\{" + ndl + r" dims = \[(.*?)\]\}", ir)
    assert match, f"no {ndl} in the converted IR:\n{ir}"
    dims = re.findall(r"(\w\(\w\)) \[(\d+), (#map\d*|-?\d+)\]", match.group(1))
    return [(tag, int(count), int(maps.get(stride, stride))) for tag, count, stride in dims]


def row_stride_bytes(memref_type):
    return (GAPPED_ROW_STRIDE if memref_type == GAPPED else ROW_SIZE) * BF16_BYTES


@pytest.mark.ci
@pytest.mark.parametrize("input_type,output_type", [
    pytest.param(GAPPED, DENSE, id="gapped-input"),
    pytest.param(DENSE, GAPPED, id="gapped-output"),
    pytest.param(GAPPED, GAPPED, id="gapped-input-and-output"),
])
def test_scalar_add_on_gapped_rows(torq_compiler, input_type, output_type):
    ir = convert_to_torqhw(torq_compiler, scalar_add_program(input_type, output_type))

    # The input is read in full vectors, one row after the other, with the input row stride.
    read = ndl_dims(ir, "DEDR")
    assert ("D(L)", VECTOR_SIZE, BF16_BYTES) in read, read
    assert ("A(H)", VECTORS_PER_ROW, VECTOR_SIZE * BF16_BYTES) in read, read
    assert ("O(H)", ROWS, row_stride_bytes(input_type)) in read, read

    # The output is written with the output row stride, and each row stops after its last
    # item, so the partial vector does not write into the gap.
    write = ndl_dims(ir, "DEQW")
    assert ("O(H)", ROWS, row_stride_bytes(output_type)) in write, write
    assert ("X(S)", ROW_SIZE, BF16_BYTES) in write, write
