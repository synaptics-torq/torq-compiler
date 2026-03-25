import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_files, chip_config
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("tosa_ops")))
def case_config(request, runtime_hw_type, chip_config):

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    failed_tc = [
            'dw_wzp.mlir' # Weight zero point support not complete
            ]

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
            'rescale-', # instable results, accuracy issues on some value (less than 1%, to be investigated)
            # error
            'asr-1x21x1024xi32',
            'matmul-in-bf16-out-fp32_207x207.mlir',
            'conv2d-f4.mlir',
            'resize-31x31x33xi8.mlir', # error: unable to free enough space for results and operand
            'conv2d-stride4-i16', # Too long to compile on next, will timeout
        ]
        if aws_fpga:
            failed_tc += [
                'add-scalar.mlir', # output mismatch
                'sub-broadcast-1.mlir', # runtime timeout
                'conv2d_bf16_2x2x4_softmax.mlir',
                'abs-int32-4.mlir',
                'rescale-rank1.mlir',
                'rescale-in-int8-out-int32-3dim.mlir',
                'argmax-1x50x256-1axis.mlir', # Number of differences: 121 out of 256 [47.27%]
                'avgpool2d.mlir',
            ]
    extra_args = {"--iree-input-type=tosa-torq":""}

    if chip_config.data['target'] != "SL2610" and (runtime_hw_type.data == "aws_fpga"):
        if request.param.data.name.startswith('mul-1hwc-in-int8'):
            pytest.xfail("Elementwise ops not supported on next chip FPGA")

    if "mul-1hwc-in-int16-out-int16" in request.param.name:
        # specific input data for the mul test case
        extra_args["tweaked_input_data_range"]  = (-32768, 32767)

    if any(s in request.param.data.name for s in failed_tc):
        pytest.xfail("output mismatch or error")

    # xfail all Elementwise_ tests on non-SL2610 aws_fpga.
    # Many of these pass intermittently but fail with mismatched elements
    # that are not consistent across runs — a failing test may pass on
    # the next run, making it hard to enumerate individual failures.
    # Since all of them have same kernel, it should be treated as a single failure until the root cause is identified and fixed.
    if chip_config.data['target'] != "SL2610" and (runtime_hw_type.data == "aws_fpga"):
        if request.param.data.name.startswith('Elementwise_'):
            pytest.xfail("Elementwise ops not supported on next chip FPGA")

    torq_tiling_tc = [
        # Channels 0 to 9 are correct, channels 10 to 18 are wrong
        'conv-343',

        # error: unable to free enough space for results and operands
        'conv2d-f4',
    ]

    if chip_config.data['target'] != "SL2610":
        torq_tiling_tc += [
            # error: unable to free enough space for results and operands
            'add-rescaled-constant',

            # Channels 0 to 55 are wrong, channels 56 to 111 are correct
            'conv-stride2',
        ]

    if any(s in request.param.data.name for s in torq_tiling_tc):
        extra_args["torq_compiler_options"]  = ["--torq-enable-torq-hl-tiling"]

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
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
