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


def dispatch_dump_after(torq_compiler, tmp_path, mlir, pass_arg):
    """Compile `mlir` and return the last IR dump of the dispatch function after
    `pass_arg`."""
    model = tmp_path / "model.mlir"
    model.write_text(mlir)
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        f"--mlir-print-ir-after={pass_arg}",
        *COMMON_FLAGS,
    ]
    print("Compiling with:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
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
    %c = tosa.conv2d %x, %w, %b, %zp, %zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x224x224x3xi8>, tensor<32x3x3x3xi8>, tensor<32xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x224x224x32xi32>
    %r = tosa.rescale %c, %m, %s, %z32, %zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x224x224x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x224x224x32xi8>
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
    %c = tosa.conv2d %x, %w, %b, %zp, %zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<1x20x20x256xi8>, tensor<512x3x3x256xi8>, tensor<512xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x10x10x512xi32>
    %r = tosa.rescale %c, %m, %s, %z32, %zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x10x10x512xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x10x10x512xi8>
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


def depthwise_block(h, cin, c, with_conv):
    """TOSA i8 block: optional 1x1 conv cin -> c with its rescale, then a 3x3
    depthwise conv on c channels with its rescale, at h x h."""
    x = "%r" if with_conv else "%x"
    conv = f"""
    %w = "tosa.const"() <{{values = dense<3> : tensor<{c}x1x1x{cin}xi8>}}> : () -> tensor<{c}x1x1x{cin}xi8>
    %c = tosa.conv2d %x, %w, %b, %zp, %zp {{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}} : (tensor<1x{h}x{h}x{cin}xi8>, tensor<{c}x1x1x{cin}xi8>, tensor<{c}xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x{h}x{h}x{c}xi32>
    %r = tosa.rescale %c, %m, %s, %z32, %zp {{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true}} : (tensor<1x{h}x{h}x{c}xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x{h}x{h}x{c}xi8>"""
    return f"""
module {{
  func.func @main(%x: tensor<1x{h}x{h}x{cin if with_conv else c}xi8>) -> tensor<1x{h}x{h}x{c}xi8> {{
    %b = "tosa.const"() <{{values = dense<7> : tensor<{c}xi32>}}> : () -> tensor<{c}xi32>
    %dw = "tosa.const"() <{{values = dense<2> : tensor<3x3x{c}x1xi8>}}> : () -> tensor<3x3x{c}x1xi8>
    %zp = "tosa.const"() <{{values = dense<0> : tensor<1xi8>}}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{{values = dense<1073741824> : tensor<1xi32>}}> : () -> tensor<1xi32>
    %s = "tosa.const"() <{{values = dense<38> : tensor<1xi8>}}> : () -> tensor<1xi8>
    %z32 = "tosa.const"() <{{values = dense<0> : tensor<1xi32>}}> : () -> tensor<1xi32>{conv if with_conv else ""}
    %d = tosa.depthwise_conv2d {x}, %dw, %b, %zp, %zp {{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}} : (tensor<1x{h}x{h}x{c}xi8>, tensor<3x3x{c}x1xi8>, tensor<{c}xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x{h}x{h}x{c}xi32>
    %y = tosa.rescale %d, %m, %s, %z32, %zp {{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true}} : (tensor<1x{h}x{h}x{c}xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x{h}x{h}x{c}xi8>
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
