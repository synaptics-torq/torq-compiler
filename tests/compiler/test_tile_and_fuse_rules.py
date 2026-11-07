import json
import re
import subprocess

import pytest

# IR checks for fusion and slicing decisions that otherwise only show up as
# compile-time profile regressions. Each test compiles a small model and reads
# one pass's IR dump for the real dispatch (not the tile-fit probe copies).

COMMON_FLAGS = [
    "--torq-hw=SL2610",
    "--torq-target-host-triple=native",
    "--torq-disable-host",
    "--torq-disable-css",
    "--mlir-disable-threading",
]


def dispatch_dump_after(torq_compiler, tmp_path, mlir, pass_arg, timeout=None, flags=COMMON_FLAGS):
    """Compile `mlir` and return the last IR dump of the dispatch function after
    `pass_arg`. `timeout` (seconds) stops a compile that hangs."""
    model = tmp_path / "model.mlir"
    model.write_text(mlir)
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        f"--mlir-print-ir-after={pass_arg}",
        *flags,
    ]
    print("Compiling with:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"

    dumps = [d for d in proc.stderr.split("// -----// IR Dump After") if "func.func @main_dispatch" in d]
    assert dumps, f"no dump of the dispatch after {pass_arg}"
    return dumps[-1]


# First layer of a CNN: 3 input channels, pad 1. Tile-and-fuse splits the 32
# output channels, so every tile reads the whole input.
CIN3_CONV = """
module {
  func.func @main(%x: tensor<1x224x224x3xi8>) -> tensor<1x224x224x32xi8> {
    %w = "tosa.const"() <{values = dense<3> : tensor<32x3x3x3xi8>}> : () -> tensor<32x3x3x3xi8>
    %b = "tosa.const"() <{values = dense<7> : tensor<32xi32>}> : () -> tensor<32xi32>
    %zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %s = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %z32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %c = tosa.conv2d %x, %w, %b, %zp, %zp pad([1, 1, 1, 1]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<1x224x224x3xi8>, tensor<32x3x3x3xi8>, tensor<32xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x224x224x32xi32>
    %r = tosa.rescale %c, %m, %s, %z32, %zp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x224x224x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x224x224x32xi8>
    return %r : tensor<1x224x224x32xi8>
  }
}
"""


@pytest.mark.ci
def test_whole_input_producers_stay_out_of_tile_loop(torq_compiler, tmp_path):
    """The conv's NHWC->NCHW transpose and pad are in its fuse group. When the
    tiles split output channels, fusing them would redo both over the whole
    input once per tile (resnet50 conv5_block2_1_conv was 2.37x main)."""
    dump = dispatch_dump_after(torq_compiler, tmp_path, CIN3_CONV, "torq-tile-and-fuse")

    loop = dump.find("scf.for")
    assert loop >= 0, f"the conv was not tiled, so this test proves nothing:\n{dump}"
    in_loop = dump[loop:]
    assert not re.search(r"linalg\.transpose ins\([^)]*tensor<1x224x224x3xi8>", in_loop), (
        f"the input transpose is recomputed in the tile loop:\n{dump}"
    )
    assert "tensor.pad" not in in_loop, f"the input pad is recomputed in the tile loop:\n{dump}"


# Two batch matmuls of the same shape. CSE gives them one zero init.
SHARED_INIT_MATMULS = """
module {
  func.func @main(%a: tensor<4x100x32xi8>, %b: tensor<4x32x100xi8>, %c: tensor<4x100x32xi8>, %d: tensor<4x32x100xi8>) -> (tensor<4x100x100xi32>, tensor<4x100x100xi32>) {
    %zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %x = tosa.matmul %a, %b, %zp, %zp : (tensor<4x100x32xi8>, tensor<4x32x100xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x100x100xi32>
    %y = tosa.matmul %c, %d, %zp, %zp : (tensor<4x100x32xi8>, tensor<4x32x100xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x100x100xi32>
    return %x, %y : tensor<4x100x100xi32>, tensor<4x100x100xi32>
  }
}
"""


@pytest.mark.ci
def test_shared_zero_init_is_filled_per_slice(torq_compiler, tmp_path):
    """A zero init that two groups share is filled in each slice. Left out of
    the groups, each tile read it from memory (yolo26s attention 2x DMA)."""
    dump = dispatch_dump_after(torq_compiler, tmp_path, SHARED_INIT_MATMULS, "torq-linalg-slicing")
    assert "scf.forall" in dump, f"nothing was sliced, so this test proves nothing:\n{dump}"
    assert not re.search(r"linalg\.fill .*outs\([^)]*tensor<4x100x100xi32>\)", dump), (
        f"the shared zero init is filled once for all tiles:\n{dump}"
    )


def in_forall(dump, op_name):
    """For each line of `dump` that contains `op_name`, True when an scf.forall
    encloses that line."""
    result = []
    open_foralls = []
    for line in dump.splitlines():
        indent = len(line) - len(line.lstrip())
        stripped = line.lstrip()
        if open_foralls and stripped.startswith("}") and indent == open_foralls[-1]:
            open_foralls.pop()
        if op_name in line:
            result.append(bool(open_foralls))
        if "scf.forall (" in line:
            open_foralls.append(indent)
    return result


# yolo26s: 3x3 stride-2 conv, 256 -> 512 channels. Its weight is too large to
# keep one copy per slice, and tile-and-fuse splits its output channels.
HEAVY_CONV = """
module {
  func.func @main(%x: tensor<1x20x20x256xi8>) -> tensor<1x10x10x512xi8> {
    %w = "tosa.const"() <{values = dense<3> : tensor<512x3x3x256xi8>}> : () -> tensor<512x3x3x256xi8>
    %b = "tosa.const"() <{values = dense<7> : tensor<512xi32>}> : () -> tensor<512xi32>
    %zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %s = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %z32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %c = tosa.conv2d %x, %w, %b, %zp, %zp pad([1, 0, 1, 0]) stride([2, 2]) dilation([1, 1]) acc_type(i32) : (tensor<1x20x20x256xi8>, tensor<512x3x3x256xi8>, tensor<512xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x10x10x512xi32>
    %r = tosa.rescale %c, %m, %s, %z32, %zp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x10x10x512xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x10x10x512xi8>
    return %r : tensor<1x10x10x512xi8>
  }
}
"""


@pytest.mark.ci
def test_heavy_conv_tiled_on_channels_uses_both_slices(torq_compiler, tmp_path):
    """Tiles that split only output channels load each weight once, so the conv
    stays dual-sliced, one 64-channel hardware tile per slice. On one slice,
    yolo26s was 3.3% slower than main."""
    dump = dispatch_dump_after(torq_compiler, tmp_path, HEAVY_CONV, "torq-linalg-slicing")
    assert "torq-single-slice" not in dump, f"the conv was kept on one slice:\n{dump}"
    assert any(in_forall(dump, "linalg.conv_2d_nchw_fchw")), f"the conv was not sliced:\n{dump}"
    sizes = re.findall(r"linalg\.conv_2d_nchw_fchw .*?-> tensor<1x(\d+)x10x10xi32>", dump)
    assert sizes and all(int(size) % 64 == 0 for size in sizes), (
        f"a slice computes a partial 64-channel hardware tile: {sizes}\n{dump}"
    )


# SiLU as torch lowers it: x * sigmoid(x). Both mul inputs are live, but they
# are functions of one value, so this is a gate, not a residual add.
SILU = """
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%x: tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16> {
    %one = arith.constant 1.0 : bf16
    %e = tensor.empty() : tensor<1x144x56x56xbf16>
    %sig = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x144x56x56xbf16>) outs(%e : tensor<1x144x56x56xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %n = arith.negf %in : bf16
      %ex = math.exp %n : bf16
      %d = arith.addf %ex, %one : bf16
      %r = arith.divf %one, %d : bf16
      linalg.yield %r : bf16
    } -> tensor<1x144x56x56xbf16>
    %y = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x, %sig : tensor<1x144x56x56xbf16>, tensor<1x144x56x56xbf16>) outs(%e : tensor<1x144x56x56xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %m = arith.mulf %a, %b : bf16
      linalg.yield %m : bf16
    } -> tensor<1x144x56x56xbf16>
    return %y : tensor<1x144x56x56xbf16>
  }
}
"""


@pytest.mark.ci
def test_silu_gate_is_sliced(torq_compiler, tmp_path):
    """A group whose two-input op gates one value must still be split across
    slices. Treating it as a residual add kept torch efficientnet_b0's SiLU
    layers on one slice, 1.5x main."""
    dump = dispatch_dump_after(torq_compiler, tmp_path, SILU, "torq-linalg-slicing")
    assert "scf.forall" in dump, f"the SiLU group was not sliced:\n{dump}"


RECIPROCAL = """
module {
  func.func @main(%x: tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16> {
    %e = tensor.empty() : tensor<1x144x56x56xbf16>
    %y = linalg.elementwise <reciprocal> ins(%x : tensor<1x144x56x56xbf16>) outs(%e : tensor<1x144x56x56xbf16>) -> tensor<1x144x56x56xbf16>
    return %y : tensor<1x144x56x56xbf16>
  }
}
"""


@pytest.mark.ci
@pytest.mark.parametrize("executor,sliced", [("nss", True), ("css", False), ("host", False)])
def test_op_assigned_off_the_slices_is_not_sliced(torq_compiler, tmp_path, executor, sliced):
    """An op that the executor map sends to CSS or the host is not split across
    the slices. Only Host ops were skipped, so a CSS op was sliced."""
    lines = RECIPROCAL.split("\n")
    line = next(i for i, text in enumerate(lines) if "linalg.elementwise <reciprocal>" in text)
    column = lines[line].index("linalg.elementwise <reciprocal>") + 1
    executor_map = tmp_path / "executor_map.json"
    executor_map.write_text(
        json.dumps({"op_assignments": {f"{line + 1}:{column}": {"executor": executor}}})
    )
    # CSS and host ops need their back ends.
    flags = [f for f in COMMON_FLAGS if not f.startswith("--torq-disable-")]
    dump = dispatch_dump_after(
        torq_compiler, tmp_path, RECIPROCAL, "torq-linalg-slicing",
        flags=[*flags, f"--torq-executor-map={executor_map}"],
    )
    assert any(in_forall(dump, "linalg.elementwise <reciprocal>")) == sliced, (
        f"the reciprocal on {executor} was {'not ' if sliced else ''}sliced:\n{dump}"
    )


SOFTMAX = """
module {
  func.func @main(%x: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    %e = tensor.empty() : tensor<1x8x64x64xbf16>
    %y = linalg.softmax dimension(3) ins(%x : tensor<1x8x64x64xbf16>) outs(%e : tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %y : tensor<1x8x64x64xbf16>
  }
}
"""


@pytest.mark.ci
def test_softmax_is_sliced_on_whole_rows(torq_compiler, tmp_path):
    """A softmax is split across the slices on a dimension other than its
    softmax dimension, so each slice reduces whole rows. On one slice, the
    Moonshine encoder softmaxes left the other slice idle."""
    dump = dispatch_dump_after(torq_compiler, tmp_path, SOFTMAX, "torq-linalg-slicing")
    assert any(in_forall(dump, "linalg.softmax")), f"the softmax was not sliced:\n{dump}"
    assert re.search(r"linalg\.softmax .*-> tensor<1x4x64x64xbf16>", dump), (
        f"the softmax was not sliced on dimension 1:\n{dump}"
    )


def transpose(in_shape, out_shape):
    return f"""
module {{
  func.func @main(%x: tensor<{in_shape}xbf16>) -> tensor<{out_shape}xbf16> {{
    %e = tensor.empty() : tensor<{out_shape}xbf16>
    %y = linalg.transpose ins(%x : tensor<{in_shape}xbf16>) outs(%e : tensor<{out_shape}xbf16>) permutation = [0, 2, 1, 3]
    return %y : tensor<{out_shape}xbf16>
  }}
}}
"""


@pytest.mark.ci
@pytest.mark.parametrize(
    "in_shape,out_shape,slices",
    [
        # No dimension splits evenly into long runs: 207 is split unevenly, 2 x 100
        # on the slices and a remainder of 7 (Moonshine encoder Transpose_114).
        ("1x8x207x36", "1x207x8x36", ["1x100x8x36", "1x7x8x36"]),
        # 16 splits evenly, so it is taken over the outer 100, which does not.
        ("1x16x100x64", "1x100x16x64", ["1x100x8x64"]),
        # 50 does not split evenly but is at least 4 chunks: 2 x 24 and a remainder of 2.
        ("1x8x50x36", "1x50x8x36", ["1x24x8x36", "1x2x8x36"]),
        # 30 does not split evenly and is too short for an uneven split.
        ("1x8x30x36", "1x30x8x36", None),
    ],
)
def test_transpose_slicing_dimension(torq_compiler, tmp_path, in_shape, out_shape, slices):
    """A transpose that keeps its innermost dimension is sliced on a dimension that
    splits evenly if there is one, and otherwise on a long uneven dimension."""
    dump = dispatch_dump_after(
        torq_compiler, tmp_path, transpose(in_shape, out_shape), "torq-linalg-slicing"
    )
    sliced = in_forall(dump, "linalg.transpose")
    if slices is None:
        assert sliced and not any(sliced), f"the transpose was sliced:\n{dump}"
        return
    assert any(sliced), f"the transpose was not sliced:\n{dump}"
    for shape in slices:
        assert re.search(rf"linalg\.transpose .*outs\([^)]*tensor<{shape}xbf16>\)", dump), (
            f"no transpose writes a {shape} slice:\n{dump}"
        )


def depthwise_block(h, cin, c, with_conv):
    """TOSA i8 block: optional 1x1 conv cin -> c with its rescale, then a 3x3
    depthwise conv on c channels with its rescale, at h x h."""
    x = "%r" if with_conv else "%x"
    conv = f"""
    %w = "tosa.const"() <{{values = dense<3> : tensor<{c}x1x1x{cin}xi8>}}> : () -> tensor<{c}x1x1x{cin}xi8>
    %c = tosa.conv2d %x, %w, %b, %zp, %zp pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<1x{h}x{h}x{cin}xi8>, tensor<{c}x1x1x{cin}xi8>, tensor<{c}xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x{h}x{h}x{c}xi32>
    %r = tosa.rescale %c, %m, %s, %z32, %zp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x{h}x{h}x{c}xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x{h}x{h}x{c}xi8>"""
    return f"""
module {{
  func.func @main(%x: tensor<1x{h}x{h}x{cin if with_conv else c}xi8>) -> tensor<1x{h}x{h}x{c}xi8> {{
    %b = "tosa.const"() <{{values = dense<7> : tensor<{c}xi32>}}> : () -> tensor<{c}xi32>
    %dw = "tosa.const"() <{{values = dense<2> : tensor<3x3x{c}x1xi8>}}> : () -> tensor<3x3x{c}x1xi8>
    %zp = "tosa.const"() <{{values = dense<0> : tensor<1xi8>}}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{{values = dense<1073741824> : tensor<1xi32>}}> : () -> tensor<1xi32>
    %s = "tosa.const"() <{{values = dense<38> : tensor<1xi8>}}> : () -> tensor<1xi8>
    %z32 = "tosa.const"() <{{values = dense<0> : tensor<1xi32>}}> : () -> tensor<1xi32>{conv if with_conv else ""}
    %d = tosa.depthwise_conv2d {x}, %dw, %b, %zp, %zp pad([1, 1, 1, 1]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<1x{h}x{h}x{c}xi8>, tensor<3x3x{c}x1xi8>, tensor<{c}xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x{h}x{h}x{c}xi32>
    %y = tosa.rescale %d, %m, %s, %z32, %zp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x{h}x{h}x{c}xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x{h}x{h}x{c}xi8>
    return %y : tensor<1x{h}x{h}x{c}xi8>
  }}
}}
"""


@pytest.mark.ci
@pytest.mark.parametrize(
    "h,cin,c",
    [
        # TileAndFuse puts the conv and the depthwise in one tile loop.
        (56, 24, 144),
        # 60 channels: the conv is sliced with a peeled remainder of 4.
        (28, 64, 60),
    ],
)
def test_depthwise_after_conv_stays_whole(torq_compiler, tmp_path, h, cin, c):
    """A depthwise conv whose input a conv in the same block computes stays on
    one slice. Slicing it on channels made MobileNet V2 slower than one slice."""
    dump = dispatch_dump_after(
        torq_compiler, tmp_path, depthwise_block(h, cin, c, True), "torq-linalg-slicing"
    )
    assert any(in_forall(dump, "linalg.conv_2d_nchw_fchw")), (
        f"the conv was not sliced, so this test proves nothing:\n{dump}"
    )
    dw = in_forall(dump, "linalg.depthwise_conv_2d_nchw_chw")
    assert dw and not any(dw), f"the depthwise conv was sliced:\n{dump}"


@pytest.mark.ci
def test_depthwise_on_tiled_input_is_sliced(torq_compiler, tmp_path):
    """A depthwise conv that reads an extract_slice of a value from outside its
    tile loop is still sliced."""
    dump = dispatch_dump_after(
        torq_compiler, tmp_path, depthwise_block(112, 0, 96, False), "torq-linalg-slicing"
    )
    assert "scf.for " in dump, f"the depthwise conv was not tiled, so this test proves nothing:\n{dump}"
    assert any(in_forall(dump, "linalg.depthwise_conv_2d_nchw_chw")), (
        f"the depthwise conv was not sliced:\n{dump}"
    )


# 1x1 conv and rescale, then the two H halves swap with slice + concat. The
# trailing NHWC transpose reads the rescale output only through the fixed
# halves, which feed tensor.insert_slice outside its tile loop (issue #2673).
CONV_SLICE_CONCAT = """
module {
  func.func @main(%x: tensor<1x120x160x16xi8>) -> tensor<1x120x160x64xi8> {
    %w = "tosa.const"() <{values = dense<1> : tensor<64x1x1x16xi8>}> : () -> tensor<64x1x1x16xi8>
    %b = "tosa.const"() <{values = dense<0> : tensor<64xi32>}> : () -> tensor<64xi32>
    %z = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %z32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %m = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %sh = "tosa.const"() <{values = dense<36> : tensor<1xi8>}> : () -> tensor<1xi8>
    %c = tosa.conv2d %x, %w, %b, %z, %z pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<1x120x160x16xi8>, tensor<64x1x1x16xi8>, tensor<64xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x120x160x64xi32>
    %r = tosa.rescale %c, %m, %sh, %z32, %z scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x120x160x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x120x160x64xi8>
    %s0 = tosa.const_shape values(dense<[0, 0, 0, 0]> : tensor<4xindex>) : () -> !tosa.shape<4>
    %s1 = tosa.const_shape values(dense<[0, 60, 0, 0]> : tensor<4xindex>) : () -> !tosa.shape<4>
    %sz = tosa.const_shape values(dense<[1, 60, 160, 64]> : tensor<4xindex>) : () -> !tosa.shape<4>
    %lo = tosa.slice %r, %s0, %sz : (tensor<1x120x160x64xi8>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x60x160x64xi8>
    %hi = tosa.slice %r, %s1, %sz : (tensor<1x120x160x64xi8>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x60x160x64xi8>
    %cat = tosa.concat %hi, %lo axis(1) : (tensor<1x60x160x64xi8>, tensor<1x60x160x64xi8>) -> tensor<1x120x160x64xi8>
    return %cat : tensor<1x120x160x64xi8>
  }
}
"""


@pytest.mark.ci
def test_fixed_slices_outside_tile_loop_are_not_charged(torq_compiler, tmp_path):
    """The LRAM fit probe charged the fixed concat halves at every tile size, so
    no tile fit and the transpose got a 1x1x1x1 tile. The compile then hung in
    UnrollLoop."""
    dump = dispatch_dump_after(
        torq_compiler, tmp_path, CONV_SLICE_CONCAT, "torq-tile-and-fuse", timeout=300
    )
    transpose_loop = (
        r"scf\.for %\w+ = %\w+ to %c120(_\d+)? step %c24(_\d+)? "
        r"iter_args\([^)]*\) -> \(tensor<1x120x160x64xi8>\)"
    )
    assert re.search(transpose_loop, dump), (
        f"the transpose is not tiled in rows of 24:\n{dump}"
    )
