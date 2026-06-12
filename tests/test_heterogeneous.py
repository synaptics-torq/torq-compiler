import pytest
from torq.testing.cases import get_test_cases_from_files
from torq.testing.iree import list_mlir_files
from torq.testing.comparison import compare_test_results
from torq.testing.cases import Case
from torq.testing.versioned_fixtures import versioned_generated_file_fixture, versioned_hashable_object_fixture, VersionedUncachedData, versioned_unhashable_object_fixture
from .test_tflite_mbv2 import download_mobilenetv2_model, mbv2_compile_options, mbv2_input_data, _compare_full_model_pair


@pytest.fixture
def torq_compiler(llvmcpu_compiler):
    return llvmcpu_compiler


@pytest.fixture
def mbv2_model(request, cache):
    return VersionedUncachedData(data=download_mobilenetv2_model(cache), version="mbv2_model")


@versioned_unhashable_object_fixture
def heterogeneous_extra_compiler_options(request, runtime_hw_type):
    if runtime_hw_type == "astra_machina":
        return ["--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu", "--iree-llvmcpu-target-cpu=generic", "--iree-llvmcpu-target-cpu-features=+neon,+crypto,+crc,+dotprod,+rdm,+rcpc,+lse,+sve"]
    else:
        return ["--iree-llvmcpu-target-cpu=host"]


def generate_cases():

    cases = []

    heterogeneous_case = {
        "torq_compiler_options": ["--iree-hal-target-device=torq=torq", "--iree-hal-target-device=local=local", 
                                "--iree-hal-local-target-device-backends=llvm-cpu",
                                "--iree-flow-disable-deduplication", "--mlir-disable-threading"],
        "torq_runtime_options": ["--device=local-task"],
        "extra_torq_compile_option": "heterogeneous_extra_compiler_options"
    }
    
    # add all the options that are required for the torq backend to work
    heterogeneous_case["torq_compiler_options"].extend(["--iree-opt-const-expr-hoisting=false", 
                                                        "--iree-flow-enable-pad-handling",
                                                        "--iree-flow-inline-constants-max-byte-length=100000000",
                                                        "--iree-dispatch-creation-enable-elementwise-fusion=false", 
                                                        "--iree-preprocessing-enable-elementwise-fusion=false"])

    test_files = [x for x in list_mlir_files("tosa_ops") if "softmax-1x1000xi8.mlir" == x.name]

    for case_data in test_files:
        case_config = heterogeneous_case.copy()

        case_config["mlir_model_file"] = "static_mlir_model_file"
        case_config["static_mlir_model_file"] = str(case_data)
        case_config["input_data"] = "tweaked_random_input_data"
        case_config["comparison_config"] = "comparison_config_from_mlir"
        case_config["affinities"] = [{'index': 13, 'affinity': 'local'}]
        case_config["reference"] = "llvmcpu_reference_results"

        cases.append(Case(case_data.name, case_config))

    mbv2_config = heterogeneous_case.copy()
    mbv2_config["tflite_model_file"] = "mbv2_model"
    mbv2_config["mlir_model_file"] = "tflite_mlir_model_file"
    mbv2_config["input_data"] = "mbv2_input_data"
    mbv2_config["torq_compiler_timeout"] = 1200
    mbv2_config["torq_runtime_timeout"] = 10 * 60
    mbv2_config["affinities"] = [{'index': 286, 'affinity': 'local'}, {'index': 287, 'affinity': 'torq'}]
    mbv2_config["reference"] = "tflite_reference_results"
    mbv2_config["comparison_func"] = "mbv2_comparison"

    cases.append(Case("mbv2_full_model", mbv2_config))

    return cases


@pytest.fixture(params=generate_cases())
def case_config(request):
    return request.param.data


def test_heterogeneous_compile(torq_compiled_model):
    pass


def test_heterogeneous_run(torq_results):
    pass


@pytest.mark.astra_machina_sl_ci
@pytest.mark.ci
def test_heterogeneous_compare(request, torq_results, case_config):

    reference = request.getfixturevalue(case_config["reference"])    
    
    if case_config.get("comparison_func") == "mbv2_comparison":
        _compare_full_model_pair(torq_results, reference, "torq", "tflite")
    else:
        compare_test_results(request, torq_results, reference, case_config)
