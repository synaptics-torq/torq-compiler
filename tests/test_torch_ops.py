import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_file_group
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture
def comparison_config_for_conv_truncf(request):
    """Comparison config for conv1d with truncf before reduce (memory-optimized mode)."""
    return {"fp_avg_tol": 0.2, "fp_max_tol": 1.0}


@pytest.fixture(params=get_test_cases_from_files(list_mlir_file_group("torch_ops")))
def case_config(request, runtime_hw_type, chip_config):

    next_chip = (chip_config.data['target'] != "SL2610")

    no_negative_input = [
        'sqrt-scalar',
    ]

    extra_args = {}
    if any(s in request.param.name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    iree_regression_tc = [
        # error: CSS program allocations of <N> bytes exceed maximum CSS stack size of ?
        "0464_Concat__layers.3_self_attn_Concat_5.mlir",
        "concat-axis-3-1x256x128x2-b16.mlir",

        # Assertion `llvm::isUIntN(BitWidth, val) && "Value is not an N-bit unsigned value"' failed.
        "softmax-1x8x1x207xbf16.mlir",

        # Failure due to increased differences
        "dw1d_k5_d2_s1.mlir",
        "dw1d_k5_d4_s1.mlir",
        "dw1d_k5_d8_s1.mlir",
        "dw1d_k5_dilation16_stride1.mlir",
        "dw_1x4_dilation_1x2_expanded_1x7.mlir",
        "reducemean.mlir",
        "reducemean-reshape.mlir",
    ]
    if next_chip:
        iree_regression_tc += [
            "decoder.mlir.431.Mul_24.mlir",
            "sqrt-scalar-bf16.mlir",
            "div_1x1x288xbf16.mlir",
            "collapse-issue-moonshine.mlir",
        ]
    if any(s in request.param.data.name for s in iree_regression_tc):
        pytest.xfail("IREE 3.10 regression failure")

    failed_tc = [
        'equal.mlir',
        'instancenorm.mlir', 
        '0135_ReduceMean__layers.0_post_attention_layernorm_ReduceMean.mlir',

        # Tracked by issue #1091
        # A big tensor.extract inside a linalg.generic causes tile-and-fuse to fail
        'ConvTranspose_bf16_1x1x512x512.mlir',

        # Tracked by issue #1092
        # LLVM ERROR: Unsupported non-tensor input during outlining
        'dynamicquantize.mlir',
    ]

    if aws_fpga:
        # aws-fpga failures
        failed_tc += [
            'decoder.mlir.431.Mul_24.mlir', # AssertionError: Number of differences: 53378 out of 59616 [89.54%]
            'add-1x24x56x56-bf16.mlir', # output is all Nan
            'gather', # DEDR producer/consumer hangs on FPGA
        ]

    if next_chip:
        # Next chip failures
        print(f"[pytest] True Running test with chip target: {chip_config.data['target']}")

        failed_tc += [
            # Max relative difference: 1.0
            # Max absolute difference: 40.0
            # Number of differences: 18432 out of 622592 [2.96%]
            'maxpool_bf16_batch19_4x1_stride4x1',

            # Max relative difference: 0.9999998211860657
            # Max absolute difference: 6.0
            # Number of differences: 1914 out of 401408 [0.48%]
            'conv2d-nchw-clip-bf16.mlir',

            # Max relative difference: 0.012047598138451576
            # Max absolute difference: 0.0009765625
            # Number of differences: 3 out of 59616 [0.01%]
            'gemm_bf16'
        ]
        if aws_fpga:
            # aws-fpga failures
            failed_tc += [
                # native_executable.cc:1085: INTERNAL; torq failed to wait;
                'encoder.mlir.237.Conv_1_small.mlir', # AssertionError: Number of differences: 53378 out of 59616 [89.54%]
                'encoder.mlir.243.Conv_2_small.mlir',
                '0698_Neg__layers.5_self_attn_Neg.mlir', # output mismatch
                '0730_Cast__layers.5_Add_output_0_cast_to_fp32.mlir', # output mismatch
                'encoder.mlir.230.Conv_0_small.mlir', # output mismatch
                'abs.mlir',  # output mismatch
                'softmax-1x8x1x207xbf16.mlir'
            ]
    else:
        print(f"[pytest] False Running test with chip target: {chip_config.data['target']}")


    if any(s in request.param.name for s in failed_tc):
        pytest.xfail("output mismatch or error")

    torq_tiling_tc = [
        # <unknown>:0: error: expected integer or index type
        # Assertion failed: (succeeded(ConcreteT::verify(getDefaultDiagnosticEmitFn(ctx), args...))), 
        #    function get, file StorageUniquerSupport.h, line 180.
        # Triggered from ConvertOddDimensionStrideConvPattern<mlir::syna::torq_hl::Conv2DOp>::matchAndRewrite()
        'conv2d-nchw-clip-bf16.mlir',

        'Conv2d_bf16_1x1x64x8192',
        'reducemean-reshape',
        'reducemean.mlir',
    ]

    if any(s in request.param.data.name for s in torq_tiling_tc):
        extra_args["torq_compiler_options"]  = ["--torq-enable-torq-hl-tiling"]

    # Option Test for conv1d with truncf before reduce (memory-optimized mode) to maintain easily
    # This enables --torq-conv1d-truncate-for-reduce to test bf16 reduce input
    if 'encoder.mlir.230.Conv_0_small.mlir' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-conv1d-truncate-for-reduce=true"]
        extra_args["comparison_config"] = "comparison_config_for_conv_truncf"

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
