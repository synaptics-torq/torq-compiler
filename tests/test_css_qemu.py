import pytest

from torq.testing.iree import MODELS_DIR
from torq.testing.comparison import compare_test_results
from .models.keras_models import conv_act_model

from torq.testing.versioned_fixtures import versioned_cached_data_fixture
from torq.testing.cases import Case

from .models.keras_models import conv_act_model


def get_test_cases():

    test_cases = []

    for name in ["matmul-notile", "softmax"]:
        test_cases.append(Case("tosa_" + name, {
            "mlir_model_file": "static_mlir_model_file",
            "static_mlir_model_file": MODELS_DIR / "tosa_ops" / (name + ".mlir")
        }))

    for name in ["tensor_pad"]:
        test_cases.append(Case("linalg_" + name, {
            "mlir_model_file": "static_mlir_model_file",
            "static_mlir_model_file": MODELS_DIR / "linalg_ops" / (name + ".mlir")
        }))

    test_cases.append(Case("keras_conv_act_model", {
        "keras_model": "conv_act_model",
        "mlir_model_file": "tflite_mlir_model_file",
        "tflite_model_file": "quantized_tflite_model_file"
    }))

    return test_cases


@pytest.fixture(params=get_test_cases())
def case_config(request):
    return {        
        "input_data": "tweaked_random_input_data",
        "torq_compiler_options": ["--torq-disable-slices", "--iree-input-type=tosa", "--torq-css-qemu"],
        **request.param.data
    }


@pytest.mark.ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):    
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
