"""IR checks of the pattern that thins the kernel-1 axes of stride (2, 2) convs
(synaptics-torq/torq-compiler-dev#2890): which convs it keeps, the slice it creates, where the
border of the image stays, and the S2D fold of a thinned conv2d in the same pass. tests/ops has the
numerical tests.
"""

import re
import subprocess
from pathlib import Path

import pytest

PASS = "torq-decompose-conv-to-space-to-depth"
CHANNELS = 8
SIZE = 32

CONV_OPS = {
    "conv2d": "linalg.conv_2d_nchw_fchw",
    "depthwise": "linalg.depthwise_conv_2d_nchw_chw",
}


def conv_ir(kind, kernel, pad_h=0, pad_w=0, channels=CHANNELS, size=SIZE, dtype="f32",
            dilations=(1, 1)):
    """A stride (2, 2) conv of a `size`x`size` image that has `pad_h` and `pad_w` border elements
    on each side of the H and W axes."""
    op = CONV_OPS[kind]
    filter_channels = f"{channels}x{channels}" if kind == "conv2d" else f"{channels}"
    padded = [size + 2 * pad_h, size + 2 * pad_w]
    out = [(n - (k - 1) * d - 1) // 2 + 1 for n, k, d in zip(padded, kernel, dilations)]
    image_type = f"tensor<1x{channels}x{size}x{size}x{dtype}>"
    padded_type = f"tensor<1x{channels}x{padded[0]}x{padded[1]}x{dtype}>"
    filter_type = f"tensor<{filter_channels}x{kernel[0]}x{kernel[1]}x{dtype}>"
    out_type = f"tensor<1x{channels}x{out[0]}x{out[1]}x{dtype}>"
    return f"""
func.func @main(%image: {image_type}) -> {out_type} {{
  %zero = arith.constant 0.0 : {dtype}
  %filter = arith.constant dense<0.5> : {filter_type}
  %padded = tensor.pad %image low[0, 0, {pad_h}, {pad_w}] high[0, 0, {pad_h}, {pad_w}] {{
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
    tensor.yield %zero : {dtype}
  }} : {image_type} to {padded_type}
  %empty = tensor.empty() : {out_type}
  %init = linalg.fill ins(%zero : {dtype}) outs(%empty : {out_type}) -> {out_type}
  %result = {op} {{dilations = dense<[{dilations[0]}, {dilations[1]}]> : vector<2xi64>,
                   strides = dense<2> : vector<2xi64>}}
      ins(%padded, %filter : {padded_type}, {filter_type}) outs(%init : {out_type}) -> {out_type}
  return %result : {out_type}
}}
"""


def run_pass(torq_compiler, source, options=()):
    optimizer = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [str(optimizer), f"--pass-pipeline=builtin.module(func.func({PASS}))", *options],
        input=source, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def slices(output):
    """The (source, sizes, steps) of each tensor.extract_slice in `output`."""
    return re.findall(
        r"tensor\.extract_slice (%\w+)\[0, 0, 0, 0\] \[([\d, ]+)\] \[([\d, ]+)\]", output
    )


@pytest.mark.ci
@pytest.mark.parametrize("kernel,steps,sizes,strides", [
    ((3, 1), "1, 1, 1, 2", "1, 8, 32, 16", r"\[2, 1\]"),
    ((1, 1), "1, 1, 2, 2", "1, 8, 16, 16", "1"),
])
def test_depthwise_kernel1_axes_are_thinned(torq_compiler, kernel, steps, sizes, strides):
    output = run_pass(torq_compiler, conv_ir("depthwise", kernel))
    assert slices(output) == [("%arg0", sizes, steps)]
    assert re.search(rf"linalg\.depthwise_conv_2d_nchw_chw.*strides = dense<{strides}>", output)


@pytest.mark.ci
@pytest.mark.parametrize("kind,kernel", [
    ("conv2d", (1, 1)),     # has its own lowering
    ("conv2d", (3, 3)),
    ("depthwise", (1, 3)),  # the depthwise stride-2 kernel takes a kernel height of 1
    ("depthwise", (3, 3)),
])
def test_other_stride2_convs_are_kept(torq_compiler, kind, kernel):
    output = run_pass(torq_compiler, conv_ir(kind, kernel))
    assert not slices(output)
    assert re.search(rf"{CONV_OPS[kind]}.*strides = dense<2>", output)


@pytest.mark.ci
def test_border_of_another_axis_stays_a_pad_of_the_slice(torq_compiler):
    """The conv lowering folds a tensor.pad into the kernel only when the pad feeds the conv."""
    output = run_pass(torq_compiler, conv_ir("depthwise", (3, 1), pad_h=1))
    assert slices(output) == [("%arg0", "1, 8, 32, 16", "1, 1, 1, 2")]
    pads = re.findall(r"tensor\.pad (%\w+) low\[([\d, ]+)\] high\[([\d, ]+)\]", output)
    assert pads == [("%extracted_slice", "0, 0, 1, 0", "0, 0, 1, 0")]


@pytest.mark.ci
def test_border_of_a_thinned_axis_is_sliced_with_the_image(torq_compiler):
    output = run_pass(torq_compiler, conv_ir("depthwise", (3, 1), pad_h=1, pad_w=1))
    assert slices(output) == [("%padded", "1, 8, 34, 17", "1, 1, 1, 2")]
    assert len(re.findall(r"tensor\.pad ", output)) == 1


def conv2d_strides(output):
    """The strides of each linalg.conv_2d_nchw_fchw in `output`."""
    return re.findall(r"linalg\.conv_2d_nchw_fchw .*strides = dense<([\d, \[\]]+)>", output)


@pytest.mark.ci
@pytest.mark.parametrize("kernel,pad_h,steps,sizes", [
    ((1, 3), 0, "1, 1, 2, 1", "1, 8, 16, 32"),
    # The S2D pattern also folds a valid stride (2, 1) conv, because the image is a slice.
    ((3, 1), 0, "1, 1, 1, 2", "1, 8, 32, 16"),
    ((3, 1), 1, "1, 1, 1, 2", "1, 8, 32, 16"),
])
def test_conv2d_kernel1_axis_is_thinned_and_folded(torq_compiler, kernel, pad_h, steps, sizes):
    """The thinned conv2d has stride (1, 2) or (2, 1). The S2D pattern folds the stride-2 axis
    into the channels, which gives a stride-1 conv."""
    output = run_pass(torq_compiler, conv_ir("conv2d", kernel, pad_h=pad_h))
    assert slices(output)[0] == ("%arg0", sizes, steps)
    assert conv2d_strides(output) == ["1"]


@pytest.mark.ci
@pytest.mark.parametrize("options", [(), ("--torq-s2d-strategy=split",)])
def test_thinned_conv2d_folds_when_the_cost_gate_prefers_the_split(torq_compiler, options):
    """A stride (1, 2) conv has no residue split: without the fold it stops on an assert in
    convAdjustPadding. In this conv the weights are larger than three times the output, so the
    cost gate prefers the split for a conv that has one."""
    source = conv_ir("conv2d", (1, 3), channels=256, size=16, dtype="bf16")
    output = run_pass(torq_compiler, source, options)
    assert conv2d_strides(output) == ["1"]


@pytest.mark.ci
def test_dilated_conv2d_is_kept(torq_compiler):
    """The S2D pattern does not fold a dilated conv, so a thinned conv would keep stride (1, 2)."""
    output = run_pass(torq_compiler, conv_ir("conv2d", (1, 3), dilations=(1, 2)))
    assert not slices(output)
    assert conv2d_strides(output) == ["2"]


@pytest.mark.ci
@pytest.mark.parametrize("kernel,pad_h,pad_w,strides", [
    # The border is in memory and the sub-kernels of the split are not 1: the split does not take
    # the conv, so the S2D pattern folds it.
    ((1, 9), 0, 4, ["1"]),
    ((9, 1), 4, 0, ["1"]),
    # No border: the cost gate prefers the split, which takes the conv in a later pass.
    ((1, 9), 0, 0, ["[1, 2]"]),
])
def test_thinned_large_kernel_conv2d_folds_when_the_split_refuses_it(torq_compiler, kernel, pad_h,
                                                                     pad_w, strides):
    source = conv_ir("conv2d", kernel, pad_h=pad_h, pad_w=pad_w, size=16, dtype="bf16")
    assert conv2d_strides(run_pass(torq_compiler, source)) == strides
