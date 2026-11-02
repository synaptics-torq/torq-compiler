import re
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR


def _tile(torq_compiler, tmp_path, fixture, mode="only-patterns", extra=()):
    output = tmp_path / f"{fixture}-{len(extra)}.mlir"
    # Use torq-compile's initialized dialect registry, but run only the isolated
    # preprocessing passes: no ONNX folding or later loop peeling can hide the
    # producer types and schedule being tested.
    passes = "torq-tile-and-fuse"
    if fixture == "conv-fallback":
        passes = "torq-mark-patterns-for-tile-and-fuse,torq-tile-and-fuse{slice-count=4}"
    result = subprocess.run(
        [
            str(torq_compiler.file_path),
            str(MODELS_DIR / "tiling" / f"{fixture}.mlir"),
            "--torq-hw=SL2610",
            "--compile-from=abi",
            "--compile-to=preprocessing",
            f"--iree-preprocessing-pass-pipeline=builtin.module(func.func({passes}))",
            f"--torq-tile-and-fuse-producers-fuse-mode={mode}",
            *extra,
            "-o", str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "failed to tile" not in result.stderr, result.stderr
    return output.read_text()


def _loop_steps(ir):
    constants = dict(re.findall(r"(%[\w]+) = arith.constant (\d+) : index", ir))
    steps = re.findall(r"scf.for %\w+ = %\w+ to %\w+ step (%\w+)", ir)
    return [int(constants[step]) for step in steps]


@pytest.mark.ci
@pytest.mark.parametrize("mode", ["max-producers", "only-patterns", "max-size"])
def test_fused_int8_weight_controls_loop_order(torq_compiler, tmp_path, mode):
    ir = _tile(torq_compiler, tmp_path, "matmul-fused-i8", mode)
    outer, inner = re.findall(r"scf.for (%\w+) =", ir)
    # The outer IV slices A's rows and the inner IV slices B's columns. Pricing
    # the live int8 source as bf16 reverses this order for this shape.
    assert f"tensor.extract_slice %arg0[{outer}, 0]" in ir
    assert f"tensor.extract_slice %arg1[0, {inner}]" in ir
    assert re.search(r"tensor.extract_slice %arg1.*tensor<800x800xi8>", ir)
    assert _loop_steps(ir) == [100, 192]


@pytest.mark.ci
@pytest.mark.parametrize("mode", ["max-producers", "only-patterns"])
def test_fused_extract_keeps_legacy_tile_shape(torq_compiler, tmp_path, mode):
    ir = _tile(torq_compiler, tmp_path, "matmul-fused-extract", mode)
    disabled = _tile(
        torq_compiler, tmp_path, "matmul-fused-extract", mode,
        extra=("--torq-disable-matmul-size-order", "--torq-disable-matmul-vector-align"),
    )
    assert "tensor.extract " in ir
    assert _loop_steps(ir) == _loop_steps(disabled)
    assert _loop_steps(ir) == [200, 89]


@pytest.mark.ci
def test_conv_fallback_preserves_spatial_minimum(torq_compiler, tmp_path):
    ir = _tile(torq_compiler, tmp_path, "conv-fallback")
    assert _loop_steps(ir) == [4, 1, 1]
    assert "tensor<1x4x1x1xbf16>" in ir


@pytest.mark.ci
def test_resident_lhs_selects_n_outer_loop_order(torq_compiler, tmp_path):
    ir = _tile(torq_compiler, tmp_path, "matmul-n-outer")
    outer, inner = re.findall(r"scf.for (%\w+) =", ir)
    # The outer IV slices B's columns and the inner IV slices A's rows.
    assert f"tensor.extract_slice %arg1[0, {outer}]" in ir
    assert f"tensor.extract_slice %arg0[{inner}, 0]" in ir
    disabled = _tile(
        torq_compiler, tmp_path, "matmul-n-outer", extra=("--torq-disable-matmul-loop-reorder",)
    )
    outer, inner = re.findall(r"scf.for (%\w+) =", disabled)
    assert f"tensor.extract_slice %arg0[{outer}, 0]" in disabled
