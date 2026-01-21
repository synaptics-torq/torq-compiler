import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_files, chip_config
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("tosa_ops")))
def case_config(request, runtime_hw_type, chip_config):

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    failed_tc = []

    if aws_fpga:
        # aws-fpga failures
        failed_tc += [
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

    if chip_config.data['target'] != "SL2610":
        # Next chip failures
        failed_tc += [
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
        if aws_fpga:
            failed_tc += [
                'add-scalar.mlir', # output mismatch
                'mul-1hwc-in-int8-out-int32-1.mlir',  # output mismatch
                'mul-1hwc-in-int8-out-int16-320x240x24.mlir',  # output mismatch
                'mul-1hwc-in-int8-out-int32-3dim.mlir',  # output mismatch
                'mul-1hw1-in-int8-out-int32.mlir',  # output mismatch
                'sub-broadcast-1.mlir', # runtime timeout
                'Elementwise_lsr-i8.mlir', # output mismatch
                'Elementwise_lsl-i16.mlir', # output mismatch
                'Elementwise_eq-i32.mlir', # output mismatch
                'argmax-1x50x256-1axis.mlir', # Number of differences: 121 out of 256 [47.27%]
            ]
    extra_args = {"--iree-input-type=tosa-torq":""}

    if "mul-1hwc-in-int16-out-int16" in request.param.name:
        # specific input data for the mul test case
        extra_args["tweaked_input_data_range"]  = (-32768, 32767)

    if any(s in request.param.data.name for s in failed_tc):
        pytest.xfail("output mismatch or error")

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        **extra_args
    }

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
