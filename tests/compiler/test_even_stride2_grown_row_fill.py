"""IR checks of the grown row that torq-even-stride2-conv-pool-inputs adds to a stride [2, 1] conv
with an odd input height (synaptics-torq/torq-compiler-dev#3038). The kernel layout pass gives such
a conv zero kernel rows that read the grown row. A float conv must have a defined value in this
row, because 0 * NaN is NaN. The simulator starts with zeroed memory: the
depthwise_conv2d_bf16_8x3x1_stride2x2_pad0_8x512x1024 case of tests/testdata/torch_ops/conv.py runs
with --torq_fill_memory=0xff to show the bug itself.
"""

import re
import subprocess
from pathlib import Path

import pytest

PASS = "torq-even-stride2-conv-pool-inputs"
CHANNELS = 8
WIDTH = 16

# element type -> (bias type, output_min, output_max)
DTYPES = {
    "bf16": (f"tensor<{CHANNELS}xf32>", -8388608, 2139095040),
    "f32": (f"tensor<{CHANNELS}xf32>", -8388608, 2139095040),
    "i8": (f"tensor<{2 * CHANNELS}xi32>", -128, 127),
}


def conv_ir(dtype, height, op="depthwise_conv2d", kernel_height=3):
    """A torq_hl `op` (depthwise_conv2d, conv2d or maxpool2d) with a `kernel_height`x1 kernel,
    stride [2, 1] and no padding on an image of `height` rows."""
    bias_type, output_min, output_max = DTYPES[dtype]
    out_height = (height - kernel_height) // 2 + 1
    input_type = f"tensor<1x{CHANNELS}x{height}x{WIDTH}x{dtype}>"
    out_type = f"tensor<1x{CHANNELS}x{out_height}x{WIDTH}x{dtype}>"
    common = (
        f"output_max = {output_max} : i32, output_min = {output_min} : i32, "
        f"pad = array<i64: 0, 0, 0, 0>, segment_output = false, stride = array<i64: 2, 1>"
    )
    if op == "maxpool2d":
        weights_type = f"tensor<1x1x1x1x{dtype}>"
        bias_type = f"tensor<{2 * CHANNELS}xi32>"
        attrs = f"input_zp = 0 : i32, kernel = array<i64: {kernel_height}, 1>, {common}"
    else:
        in_channels = 1 if op == "depthwise_conv2d" else CHANNELS
        groups = CHANNELS if op == "depthwise_conv2d" else 1
        weights_type = f"tensor<{CHANNELS}x{in_channels}x{kernel_height}x1x{dtype}>"
        dw_attr = "is_dw1d_stride1 = false, " if op == "depthwise_conv2d" else ""
        attrs = (
            f"dilation = array<i64: 1, 1>, groups = {groups} : i32, input_zp = 0 : i32, "
            f"{dw_attr}nhwc_input = false, output_zp = 0 : i32, shift_factor = 0 : i32, "
            f"vectorization_mode = 0 : i32, weight_zp = 0 : i32, {common}"
        )
    return f"""
func.func @main(%input: {input_type}, %weights: {weights_type}, %bias: {bias_type}) -> {out_type} {{
  %init = tensor.empty() : {out_type}
  %result = "torq_hl.{op}"(%init, %weights, %bias, %input) <{{{attrs}}}>
      : ({out_type}, {weights_type}, {bias_type}, {input_type}) -> {out_type}
  return %result : {out_type}
}}
"""


def run_pass(torq_compiler, source):
    optimizer = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [str(optimizer), f"--pass-pipeline=builtin.module(func.func({PASS}))"],
        input=source, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def constant_rows(output):
    """The name and the shape of each splat tensor constant in `output`."""
    return re.findall(r"(%\w+) = arith\.constant dense<0\.0+e\+00> : tensor<([\dx]+)x\w+>", output)


def inserts(output):
    """The (source, destination, offsets, sizes) of each tensor.insert_slice in `output`."""
    return re.findall(
        r"tensor\.insert_slice (%\w+) into (%\w+)\[([\d, ]+)\] \[([\d, ]+)\]", output
    )


@pytest.mark.ci
@pytest.mark.parametrize("op", ["depthwise_conv2d", "conv2d"])
@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize(
    "kernel_height, row, input_row",
    [
        (3, 7, 0),  # the grown row is appended after the input
        (5, 0, 1),  # the VALID to SAME shift prepends the grown row before the input
    ],
)
def test_float_conv_writes_only_the_grown_row(torq_compiler, op, dtype, kernel_height, row,
                                              input_row):
    output = run_pass(torq_compiler, conv_ir(dtype, height=7, op=op, kernel_height=kernel_height))
    # The row goes into the tensor.empty buffer before the input. KeepConcatInLram keeps such a
    # chain in LRAM. A linalg.fill of the buffer moves the insert of the input to XRAM.
    assert "linalg.fill" not in output
    empty = re.search(rf"(%\w+) = tensor\.empty\(\) : tensor<1x8x8x16x{dtype}>", output).group(1)
    (row_const, row_shape), = constant_rows(output)
    assert row_shape == "1x8x1x16"
    row_insert, input_insert = inserts(output)
    assert row_insert == (row_const, empty, f"0, 0, {row}, 0", "1, 8, 1, 16")
    assert input_insert[0] == "%arg0"
    assert input_insert[2:] == (f"0, 0, {input_row}, 0", "1, 8, 7, 16")
    assert re.search(rf"torq_hl\.{op}.*tensor<1x8x8x16x{dtype}>\) ->", output)


@pytest.mark.ci
def test_max_pool_leaves_the_grown_row(torq_compiler):
    # A max pool gets no zero kernel row, thus it does not read the grown row.
    output = run_pass(torq_compiler, conv_ir("bf16", height=7, op="maxpool2d"))
    assert "linalg.fill" not in output and not constant_rows(output)
    assert [insert[2:] for insert in inserts(output)] == [("0, 0, 0, 0", "1, 8, 7, 16")]


@pytest.mark.ci
def test_integer_conv_leaves_the_grown_row(torq_compiler):
    output = run_pass(torq_compiler, conv_ir("i8", height=7))
    assert "linalg.fill" not in output and "arith.constant" not in output
    assert [insert[2:] for insert in inserts(output)] == [("0, 0, 0, 0", "1, 8, 7, 16")]


@pytest.mark.ci
def test_even_height_does_not_grow(torq_compiler):
    output = run_pass(torq_compiler, conv_ir("bf16", height=8))
    assert "tensor.insert_slice" not in output and "arith.constant" not in output
