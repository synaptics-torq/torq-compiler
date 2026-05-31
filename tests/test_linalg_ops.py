import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.cases import get_test_cases_from_files
from torq.testing.iree import list_mlir_files


# Cases that genuinely need the host/CSS fallback (e.g. an unsupported
# linalg.generic in the function body) live under linalg_ops_host_css/ and are
# exempted from the NSS-only forcing applied to every other linalg_ops case.
HOST_CSS_DIR = "linalg_ops_host_css"


@pytest.fixture(params=get_test_cases_from_files(
    list_mlir_files("linalg_ops") + list_mlir_files(HOST_CSS_DIR)
))
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

    # Tests that mix linalg and tosa dialects (e.g. tosa.apply_scale inside a
    # linalg.generic) need an explicit input type; auto-detection rejects the
    # mixture.
    need_input_type_tc = ["conv1d-matmul-fc-i8-bias-requant.mlir"]
    # Prevent torq-run-module timing out on aws_fpga with specific testcases
    if aws_fpga:
        need_input_type_tc += ["reshape-collapse-expand.mlir"]

    extra_args["torq_compiler_options"] = []
    if "exp.mlir" in request.param.name:
        extra_args["torq_compiler_options"].append("--torq-enable-general-exp")
    if "reciprocal-inf" in request.param.name:
        extra_args["torq_compiler_options"].append("--torq-enable-reciprocal-inf")
    if any(s in request.param.data.name for s in need_input_type_tc):
        extra_args["torq_compiler_options"].append("--iree-input-type=linalg-torq")
    # Force every linalg_ops case onto the NSS/slice path so we fail loudly if a
    # future change silently routes it to the host/CSS fallback. Cases that
    # legitimately require host/CSS live under linalg_ops_host_css/ and are
    # exempted.
    if request.param.data.parent.name != HOST_CSS_DIR:
        extra_args["torq_compiler_options"].extend([
            "--torq-disable-host", "--torq-disable-css",
        ])

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        **extra_args
    }

def _is_bf16_matmul_case(case_config):
    mlir_file = case_config.get("static_mlir_model_file")
    return mlir_file is not None and "matmul" in mlir_file.name.lower() and "bf16" in mlir_file.name.lower()


@pytest.fixture
def reference_results(request, case_config):
    if _is_bf16_matmul_case(case_config):
        return request.getfixturevalue("numpy_matmul_reference_results")
    return request.getfixturevalue("llvmcpu_reference_results")

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, reference_results, case_config):
    compare_test_results(request, torq_results, reference_results, case_config)
