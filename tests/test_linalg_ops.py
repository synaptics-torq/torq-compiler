import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.cases import get_test_cases_from_files
from torq.testing.iree import list_mlir_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("linalg_ops")))
def case_config(request, runtime_hw_type, chip_config):
    
    no_negative_input = [
        'sqrt',
    ]

    extra_args = {}
    if any(s in request.param.name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    if "trunci-i16-to-i8" in request.param.name:
        # increase input range to check out of bound cases
        extra_args["tweaked_input_data_range"]  = (-1000, 1000)

    failed_tc = []

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    # SL2610 aws-fpga failures
    if aws_fpga:
        failed_tc += [
            'select-bf16.mlir', # AssertionError: Nans differ
            'select-f32-local-scalar.mlir', # AssertionError: Number of differences: 1024 out of 1024 [100.00%]
            'select-f32.mlir', # AssertionError: Nans differ
            'select-i16-scalar.mlir', # AssertionError: Output is 0 always
            'select-i16.mlir', # AssertionError: Number of differences: 985 out of 1024 [96.19%]
            'select-i32.mlir', # AssertionError: Number of differences: 982 out of 1024 [95.90%]
            'select-i8-scalar.mlir', # AssertionError: Number of differences: 1024 out of 1024 [100.00%]
            'select-i8.mlir' #  AssertionError: Number of differences: 989 out of 1024 [96.58%]
        ]

    # Next chip failures
    if chip_config.data['target'] != "SL2610":
        failed_tc += [
            'quantized_batch_matmul.mlir',
            'reciprocal-bf16.mlir'
        ]
        if aws_fpga:
            failed_tc += [
                'dot-in-int16-out-int16.mlir', # output mismatch
                'dot-in-int8-out-int16.mlir', # AssertionError: Output is 0 always
                'matvec-in-int8-out-int16-1.mlir', # AssertionError: Number of differences: 1 out of 1 [100.00%]
                'matvec-in-int8-out-int16.mlir', # runtime timeout
                'matvec-in-int16-out-int16.mlir', # runtime timeout
                'cmpf-bf16-to-fp32.mlir', # AssertionError: Number of differences: 21 out of 256 [8.20%]
                'batch-matmul-in-int8-out-int16.mlir', # Number of differences: 8190 out of 16384 [49.99%]
                'matmul-in-int8-out-int16.mlir', # output mismatch
            ]

    if any(s in request.param.name for s in failed_tc):
        pytest.xfail("output mismatch or error")

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        "torq_compiler_options": ["--iree-input-type=linalg-torq"],
        **extra_args
    }

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
