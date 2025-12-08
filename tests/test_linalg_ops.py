import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.cases import get_test_cases_from_files
from torq.testing.iree import list_mlir_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("linalg_ops")))
def case_config(request, runtime_hw_type, chip_config):
    
    cmodel_failed_tc = [
        'rsqrt-bf16.mlir',
    ]
    if request.param.data.name in cmodel_failed_tc:
        pytest.xfail("output mismatch")

    # Next chip failures
    next_chip_failed_tc = [
        'quantized_batch_matmul.mlir',
        'reciprocal-bf16.mlir'
    ]
    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip and any(s in request.param.name for s in next_chip_failed_tc):
        pytest.xfail("output mismatch or error on next chip")

    aws_fpga = (runtime_hw_type.data == "aws_fpga")
    # SL2610 aws-fpga failures
    aws_fpga_failed_tc = [
        'select-bf16.mlir', # AssertionError: Nans differ
        'select-f32-local-scalar.mlir', # AssertionError: Number of differences: 1024 out of 1024 [100.00%]
        'select-f32.mlir', # AssertionError: Nans differ
        'select-i16-scalar.mlir', # AssertionError: Output is 0 always
        'select-i16.mlir', # AssertionError: Number of differences: 985 out of 1024 [96.19%]
        'select-i32.mlir', # AssertionError: Number of differences: 982 out of 1024 [95.90%]
        'select-i8-scalar.mlir', # AssertionError: Number of differences: 1024 out of 1024 [100.00%]
        'select-i8.mlir' #  AssertionError: Number of differences: 989 out of 1024 [96.58%]
    ]
    if aws_fpga and any(s in request.param.name for s in aws_fpga_failed_tc):
        pytest.xfail("output mismatch")

    # Next chip aws-fpga failures
    next_aws_fpga_failed_tc = [
        'dot-in-int16-out-int16.mlir',
    ]
    if next_chip and aws_fpga and any(s in request.param.name for s in next_aws_fpga_failed_tc):
        pytest.xfail("output mismatch")

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        "torq_compiler_options": ["--iree-input-type=linalg-torq"]
    }

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
