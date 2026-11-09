"""ConstantEvaluator must compute the same compile-time constants as the JIT: each model must
compile to the same vmfb with and without --torq-disable-constant-evaluator. The cases cover
the weight-preparation cones the evaluator handles: stride-2 conv and depthwise weights (slices,
transposes, loops), weight sums for the input zero point (integer reductions), LUTs and bf16."""

import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

CASES = [
    "conv_s2_rgb_input_pad_320x320x3-i8.mlir",
    "dw_3x5_stride2_mixed_phase_i8.mlir",
    "dw_4x4_stride2_pad21_11x11x8.mlir",
    "conv2d_1x1_const_weight_input_zp_matmul_as_conv-i8.mlir",
    "precalc-i8-table-per-channel.mlir",
    "mbv2-softmax-i8.mlir",
    "conv_s2_dw_w_subview-bf16.mlir",
]


def compile_model(torq_compiler, model, hw, output, extra_options):
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(output),
        f"--torq-hw={hw}",
        "--torq-target-host-triple=native",
        *extra_options,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"
    return output.read_bytes()


@pytest.mark.ci
@pytest.mark.parametrize("hw", ["SL2610", "SR200"])
@pytest.mark.parametrize("case", CASES)
def test_constant_evaluator_matches_jit(torq_compiler, tmp_path, case, hw):
    model = MODELS_DIR / "tosa_ops" / case
    evaluated = compile_model(torq_compiler, model, hw, tmp_path / "evaluated.vmfb", [])
    jit = compile_model(
        torq_compiler, model, hw, tmp_path / "jit.vmfb", ["--torq-disable-constant-evaluator"]
    )
    assert evaluated == jit, "the constants computed without the JIT differ from the JIT's"
