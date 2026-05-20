import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_files, chip_config
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("tosa_ops")))
def case_config(request, runtime_hw_type, chip_config):

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    # Prevent torq-run-module timing out on aws_fpga with specific testcases
    need_input_type_tc = []
    if aws_fpga:
        need_input_type_tc += ["pad_TL_8x8x4.mlir"]

    extra_args = {}

    if "mul-1hwc-in-int16-out-int16" in request.param.name:
        # specific input data for the mul test case
        extra_args["tweaked_input_data_range"]  = (-32768, 32767)

    extra_args["torq_compiler_options"] = []
    if any(s in request.param.data.name for s in need_input_type_tc):
        extra_args["torq_compiler_options"].append("--iree-input-type=tosa-torq")

    if "identity-" in request.param.name:
        # Identity op is expected to be a no-op, we need to enforce conversion to TorqHL to see
        # the benefits of the pattern and also to test some special features of the ALU and ACT blocks.
        extra_args["torq_compiler_options"].append("--torq-enable-tosa-identity=true")

    # Note: "yolov8_block_mul_rescale" might need "--torq-tile-and-fuse-producers-fuse-mode=max-producers"

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
