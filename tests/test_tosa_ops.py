from fnmatch import fnmatchcase

import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import MODELS_DIR, list_mlir_files
from torq.testing.cases import get_test_cases_from_files


# Cases that genuinely need the host/CSS fallback live under tosa_ops_host_css/
# and are exempted from the NSS-only forcing applied to every other tosa_ops case.
HOST_CSS_DIR = "tosa_ops_host_css"

# Settings of single MLIR files: {file name without .mlir, or a glob pattern: settings}.
# torq_compiler_options are added to those of the case; the settings under "aws_fpga" apply on the
# AWS FPGA only. Every key must match a file.
CONFIG = {
    # Single-slice tile-and-fuse must W-tile the conv+depthwise group so the depthwise input
    # becomes a W-subview of the producer tile (the regression this test guards); with slicing
    # the work is split before tiling and the subview never forms.
    "conv_s2_dw_w_subview-bf16": {"torq_compiler_options": ["--torq-disable-slicing=true"]},
    # Compiles in well under a second locally, but a CI run on sr200 (next.group) stalled past
    # the 300s default compile budget. Double it so one-off runner stalls don't red the lane.
    "gather-i16": {"torq_compiler_timeout": 600},
    # Full-range int16 products overflow the 33-bit ACT input over the 113/128-long reduction;
    # keep K * max^2 below 2^31.
    "matmul-in-int16-out-int*": {"tweaked_input_data_range": (-4096, 4096)},
    "matmul-notile-m3-k113-n80-i16": {"tweaked_input_data_range": (-4096, 4096)},
    # Prevent torq-run-module timing out on aws_fpga.
    "pad_TL_8x8x4": {"aws_fpga": {"torq_compiler_options": ["--iree-input-type=tosa-torq"]}},
    # Identity op is expected to be a no-op, we need to enforce conversion to TorqHL to see the
    # benefits of the pattern and also to test some special features of the ALU and ACT blocks.
    "identity-*": {"torq_compiler_options": ["--torq-enable-tosa-identity=true"]},
    # Note: "yolov8_block_mul_rescale" might need
    # "--torq-tile-and-fuse-producers-fuse-mode=max-producers"
}

CASES = get_test_cases_from_files(list_mlir_files("tosa_ops") + list_mlir_files(HOST_CSS_DIR))

_unused = [k for k in CONFIG if not any(fnmatchcase(c.data.stem, k) for c in CASES)]
if _unused:
    raise ValueError(f"test_tosa_ops CONFIG: no MLIR file matches {_unused}")


def _file_config(stem, runtime_hw_type):
    """The settings of CONFIG for the file stem, on runtime_hw_type ("sim", "aws_fpga", ...)."""
    config = {"torq_compiler_options": []}
    for key, settings in CONFIG.items():
        if not fnmatchcase(stem, key):
            continue
        for scope in (settings, settings.get(runtime_hw_type, {})):
            for name, value in scope.items():
                if name == "torq_compiler_options":
                    config[name] += value
                elif name != "aws_fpga":
                    config[name] = value
    return config


@pytest.fixture(params=CASES)
def case_config(request, runtime_hw_type, chip_config):

    extra_args = _file_config(request.param.data.stem, runtime_hw_type.data)

    # Force in-tree tosa_ops cases onto the NSS/slice path so we fail loudly
    # if a future change silently routes them to the host/CSS fallback.
    # tosa_ops_host_css/ and extras/tests/testdata/tosa_ops are exempted
    # (parent.name == "tosa_ops" also matches extras).
    if request.param.data.parent.resolve() == (MODELS_DIR / "tosa_ops").resolve():
        extra_args["torq_compiler_options"].extend([
            "--torq-disable-host", "--torq-disable-css",
        ])

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        **extra_args
    }

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files_torq(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)


def test_mlir_files_llvmcpu(request, llvmcpu_reference_results, case_config):
    # this test just makes sure we can create llvmcpu_reference_results for the test case
    pass
