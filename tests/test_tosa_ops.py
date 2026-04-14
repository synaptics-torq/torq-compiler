import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_files, chip_config
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("tosa_ops")))
def case_config(request, runtime_hw_type, chip_config):

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    iree_regression_tc = [
        # error: expected element type to be 'i<N>'
        "conv2d-16b-stride1-8x8x2.mlir",
        "conv2d-stride4-i16.mlir",
        "conv2d_f4_s4_64x64x16_i16.mlir",
        "conv2d_i16_f5_s2_32x32x4_o4.mlir",
        "conv2d_i16_s2_8x8x2_o4.mlir",
        "conv2d_noalign_channel_32x32x128xi16.mlir",
        "dw_i16_8x8x1.mlir",
        "dw_i16_8x8x4.mlir",
        "dw_i16_f7_8x8x1.mlir",
        "dw_i16_f7_s2_8x8x1.mlir",
        "dw_i16_s2_64x64x8.mlir",
        "dw_i16_s2_8x8x1.mlir",
        "dw_i16_s2_8x8x4.mlir",

        #error: failed to legalize unresolved materialization from ('i<N>') to ('i<N>') that remained live after conversion
        "efficientnet-sigmoid-rescale-mul.mlir",

        # error: 'tosa.const' op requires attribute 'values'
        "conv-343.mlir",
        "conv2d-28x28x512.mlir",
        "conv2d_as_fully_connected.mlir",
        "conv2d_f3_12x12x64_i16.mlir",
        "conv2d_f5_s2_64x64x16_i16.mlir",
        "dw-32x8.mlir",
        "dw_4x4_stride2.mlir",
        "maxpool2d-stride2-k3x3-pad-112x112x127.mlir",
        "pw-16x16.mlir",
        "pw-32x8-7x7x320.mlir",
        "pw-32x8.mlir",
        "yolo-table-6.mlir",

        # Assertion `llvm::isUIntN(BitWidth, val) && "Value is not an N-bit unsigned value"' failed.
        "conv2d_bf16_2x2x4_softmax.mlir",

        # error: operand #<N> does not dominate this use
        "conv2d-matmul-wzp.mlir",

        # Assertion `inputType.getRank() == permutation.size()' failed.
        "dw_16x16-bf16.mlir",
        "dw_f5_16x16-bf16.mlir",

        # error: custom op 'tosa.fully_connected' is unknown
        "fc.mlir",

        # Assertion `!empty()' failed.
        "pw-stride2.mlir",

        # error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
        "scatter.mlir",

        # Assertion `input.shape().size() + dimensions.size() == output.shape().size()' failed.
        "conv2d_f127_s64_1x16000_o250.mlir",

        # Assertion failed: (P.type1.getElementType() == P.type2.getElementType() && "Input types must match"), function prepareParams, file ConversionUtils.h, line 505.
        "sub-rescale-scalar.mlir",
        "sub-rescale-scalar-bis.mlir",

        # failed with differences
        "conv2d_f8_s4_1x1024_o256.mlir",
        "concat-a.mlir",
        "concat-b.mlir",
    ]
    if any(s in request.param.data.name for s in iree_regression_tc):
        pytest.xfail("IREE 3.10 regression failure")

    failed_tc = [
            # Tracked by #1095
            'dw_wzp.mlir', # Weight zero point support not complete

            # results mismatch
            # looking at the log you can see there's an error that is being ignored:
            # error: multiplier unexpected size:21 while tiling
            # probably related to issue #996
            'dw_f32x32_o21_i16.mlir',
            'dw_f32x32_o21_i8.mlir',
            'cast-i16-to-bf16.mlir', 'cast-i32-to-bf16.mlir', 'cast-i8-to-bf16.mlir', # see issue #1037
            'sub-rescale-tensor-int8.mlir', # 
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

            # Tracked by issue #996
            'conv2d_noalign_channel_32x32x128xi8.mlir',
            'pw-16x16.mlir',
            'pw-32x8.mlir',

            # Tracked by issue #1092
            'pw-stride2.mlir',
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

    extra_args["torq_compiler_options"] = []
    if any(s in request.param.data.name for s in torq_tiling_tc):
        extra_args["torq_compiler_options"].append("--torq-enable-torq-hl-tiling")
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
