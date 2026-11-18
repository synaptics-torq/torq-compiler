"""Helpers of the IR tests of the patterns that change the strides of the NCHW convs in the
torq-decompose-conv-to-space-to-depth pass.
"""

import re
import subprocess
from pathlib import Path

PASS = "torq-decompose-conv-to-space-to-depth"
CHANNELS = 8
SIZE = 32

CONV_OPS = {
    "conv2d": "linalg.conv_2d_nchw_fchw",
    "depthwise": "linalg.depthwise_conv_2d_nchw_chw",
}


def _pair(value):
    """`value` as a pair: one value applies to the two elements."""
    return (value, value) if isinstance(value, int) else tuple(value)


def conv_ir(kind, kernel, pad_h=0, pad_w=0, channels=CHANNELS, size=SIZE, dtype="f32",
            dilations=(1, 1), strides=(2, 2)):
    """A conv of an image of `size`: (H, W), or one value for the two axes. `pad_h` and `pad_w`
    are the border elements of the H and W axes: (low, high), or one value for the two sides."""
    op = CONV_OPS[kind]
    filter_channels = f"{channels}x{channels}" if kind == "conv2d" else f"{channels}"
    size = _pair(size)
    pads = [_pair(pad_h), _pair(pad_w)]
    padded = [n + low + high for n, (low, high) in zip(size, pads)]
    out = [(n - (k - 1) * d - 1) // s + 1
           for n, k, d, s in zip(padded, kernel, dilations, strides)]
    image_type = f"tensor<1x{channels}x{size[0]}x{size[1]}x{dtype}>"
    padded_type = f"tensor<1x{channels}x{padded[0]}x{padded[1]}x{dtype}>"
    filter_type = f"tensor<{filter_channels}x{kernel[0]}x{kernel[1]}x{dtype}>"
    out_type = f"tensor<1x{channels}x{out[0]}x{out[1]}x{dtype}>"
    low = f"0, 0, {pads[0][0]}, {pads[1][0]}"
    high = f"0, 0, {pads[0][1]}, {pads[1][1]}"
    return f"""
func.func @main(%image: {image_type}) -> {out_type} {{
  %zero = arith.constant 0.0 : {dtype}
  %filter = arith.constant dense<0.5> : {filter_type}
  %padded = tensor.pad %image low[{low}] high[{high}] {{
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
    tensor.yield %zero : {dtype}
  }} : {image_type} to {padded_type}
  %empty = tensor.empty() : {out_type}
  %init = linalg.fill ins(%zero : {dtype}) outs(%empty : {out_type}) -> {out_type}
  %result = {op} {{dilations = dense<[{dilations[0]}, {dilations[1]}]> : vector<2xi64>,
                   strides = dense<[{strides[0]}, {strides[1]}]> : vector<2xi64>}}
      ins(%padded, %filter : {padded_type}, {filter_type}) outs(%init : {out_type}) -> {out_type}
  return %result : {out_type}
}}
"""


def run_pass(torq_compiler, source, options=(), then=()):
    """Runs the pass on `source`, and after it the passes of `then`."""
    optimizer = Path(torq_compiler.file_path).parent / "iree-opt"
    passes = ",".join((PASS, *then))
    proc = subprocess.run(
        [str(optimizer), f"--pass-pipeline=builtin.module(func.func({passes}))", *options],
        input=source, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def slices(output):
    """The (source, sizes, steps) of each tensor.extract_slice in `output`."""
    return re.findall(
        r"tensor\.extract_slice (%\w+)\[0, 0, 0, 0\] \[([\d, ]+)\] \[([\d, ]+)\]", output
    )


def pads(output):
    """The (source, low, high) of each tensor.pad in `output`."""
    return re.findall(r"tensor\.pad (%\w+) low\[([\d, ]+)\] high\[([\d, ]+)\]", output)
