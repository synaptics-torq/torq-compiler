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

    # Prevent torq-run-module timing out on aws_fpga with specific testcases
    need_input_type_tc = []
    if aws_fpga:
        need_input_type_tc += ["reshape-collapse-expand.mlir"]

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
