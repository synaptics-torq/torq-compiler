import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_files, chip_config
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("tosa_ops")))
def case_config(request, runtime_hw_type, chip_config):

    # Next chip failures
    next_chip_failed_tc = [
        # failed: output mismatch
        'conv-343.mlir',
        'conv2d_f5_s2_64x64x16_i16.mlir',
        # error
        'asr-i32.mlir',
        'add-strided-scalar-i32.mlir',
        'sub-rescale-scalar.mlir', # error: Frame size is too large to fit in LRAM
        'matmul-in-bf16-out-fp32_207x207.mlir',
        'add-constant.mlir',
        'conv2d-f4.mlir',
        'resize-31x31x33xi8.mlir', # error: unable to free enough space for results and operand
    ]
    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip and any(s in request.param.data.name for s in next_chip_failed_tc):
        pytest.xfail("output mismatch or error on next chip")

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    # aws-fpga failures
    aws_fpga_failed_tc = [
        'add-32x32x128xi16.mlir', # AssertionError: Number of differences: 110543 out of 131072 [84.34%]
        'add-bf16.mlir', # AssertionError: Nans differ
        'add-broadcast.mlir', # AssertionError: Number of differences: 10060 out of 21504 [46.78%]
        'add-constant.mlir', # AssertionError: Number of differences: 10060 out of 21504 [46.78%]
        'add-i32-3d.mlir', # AssertionError: Number of differences: 9975 out of 21504 [46.39%]
        'add-i32.mlir', # AssertionError: Number of differences: 19072 out of 75264 [25.34%]
        'add-strided-scalar-i32.mlir', # AssertionError: Number of differences: 10060 out of 21504 [46.78%]
        'sub-bf16.mlir', # AssertionError: Nans differ
        'sub-broadcast.mlir', # AssertionError: Number of differences: 10065 out of 21504 [46.81%]
        'sub-i32.mlir' # AssertionError: Nans differ
    ]
    if aws_fpga and any(s in request.param.name for s in aws_fpga_failed_tc):
        pytest.xfail("output mismatch")

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        "torq_compiler_options": ["--iree-input-type=tosa-torq"]
    }

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
