import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_files, chip_config
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("tosa_ops")))
def case_config(request, runtime_hw_type, chip_config):

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    failed_tc = []

    if chip_config.data['target'] != "SL2610":
        # Next chip failures
        failed_tc += [
            # # failed: output mismatch
            'conv2d_f5_s2_64x64x16_i16.mlir',
            'sub-rescale-scalar.mlir',
            'sub-rescale-scalar-bis.mlir',
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

    # Prevent torq-run-module timing out on aws_fpga with specific testcases
    need_input_type_tc = []
    if aws_fpga:
        need_input_type_tc += ["pad_TL_8x8x4.mlir"]

    extra_args = {}

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

    extra_args["torq_compiler_options"] = []
    if any(s in request.param.data.name for s in need_input_type_tc):
        extra_args["torq_compiler_options"].append("--iree-input-type=tosa-torq")

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
