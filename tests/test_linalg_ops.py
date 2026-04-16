import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.cases import get_test_cases_from_files
from torq.testing.iree import list_mlir_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("linalg_ops")))
def case_config(request, runtime_hw_type, chip_config):

    next_chip = (chip_config.data['target'] != "SL2610")
    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    no_negative_input = [
        'sqrt',
    ]

    extra_args = {}
    if any(s in request.param.name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    if "trunci-i16-to-i8" in request.param.name:
        # increase input range to check out of bound cases
        extra_args["tweaked_input_data_range"]  = (-1000, 1000)

    # Skip DMA throughput test cases - they require --torq-fake-reduce and are
    # meant to be run only via test_dma_throughput.py.
    if "reducesum-dma-throughput-test" in request.param.name:
        pytest.skip("DMA throughput test file; run via test_dma_throughput.py")

    iree_regression_tc = [
        # Assertion `llvm::isUIntN(BitWidth, val) && "Value is not an N-bit unsigned value"' failed.
        "softmax-1x1024x64xbf16.mlir",

        # error: operand #<N> does not dominate this use
        "quantized_batch_matmul.mlir",

        # Assertion `dims.bn==<N> || dims.bn==<N> || dims.bn==<N>' failed.
        "exp.mlir",
        "sigmoid.mlir",

        # error: 'iree_tensor_ext.dispatch.tensor.store' op operand #<N> must be dispatch.tensor, but got '!iree_tensor_ext.dispatch.tensor<readonly:tensor<<N>xbf<N>>>'
        "erf-bf16.mlir",
    ]

    if aws_fpga:
        iree_regression_tc += [
            # Runtime timeout (20s)
            "extui-i8-to-i16.mlir",
        ]

    if next_chip:
        iree_regression_tc += [
            "divf-bf16.mlir",
            "pow-2.mlir",
            "sqrt-1024x64-bf16.mlir",
            "sqrt-scalar-bf16.mlir",
            "rsqrt-bf16.mlir",
            "tanh-bf16.mlir"
        ]

    if any(s in request.param.name for s in iree_regression_tc):
        pytest.xfail("IREE 3.10 regression failure")

    failed_tc = []

    # SL2610 aws-fpga failures
    if aws_fpga:
        failed_tc += [
            # DEDR producer/consumer hangs on FPGA
            'gather',
            # output mistmatch, see https://github.com/synaptics-torq/torq-compiler-dev/issues/813
            'sigmoid.mlir'
        ]

    # Next chip failures
    if next_chip:
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
                'asr-i8-scalar.mlir', # output mismatch
                'reduceproduct-a3-bf16.mlir',
                'softmax-1x1024x64xbf16.mlir',
                'Elementwise-less-than-equal-u16.mlir', # 0.02% mismatch
            ]

    # Prevent torq-run-module timing out on aws_fpga with specific testcases
    need_input_type_tc = []
    if aws_fpga:
        need_input_type_tc += ["reshape-collapse-expand.mlir"]

    if any(s in request.param.name for s in failed_tc):
        pytest.xfail("output mismatch or error")

    extra_args["torq_compiler_options"] = []
    if "exp.mlir" in request.param.name:
        extra_args["torq_compiler_options"].append("--torq-enable-general-exp")
    if "reciprocal-inf" in request.param.name:
        extra_args["torq_compiler_options"].append("--torq-enable-reciprocal-inf")
    if any(s in request.param.data.name for s in need_input_type_tc):
        extra_args["torq_compiler_options"].append("--iree-input-type=linalg-torq")

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
