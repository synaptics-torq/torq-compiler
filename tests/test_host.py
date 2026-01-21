import pytest
import numpy as np

from torq.testing.iree import MODELS_DIR
from torq.testing.comparison import compare_test_results
from .models.keras_models import conv_act_model

from torq.testing.versioned_fixtures import versioned_cached_data_fixture
from torq.testing.cases import Case


@versioned_cached_data_fixture
def equal_input_data(request):

    input_tensor = np.zeros((1, 1, 30, 1), dtype=np.int64)
    
    input_tensor[0, 0, 10, 0] = 10
    input_tensor[0, 0, 20, 0] = 20

    return [input_tensor]


def get_test_cases():

    test_cases = []

    for name in [("tosa_ops", "matmul-notile"),
                 ("tosa_ops", "softmax-1x1000xi8"),
                 ("arith_ops", "trunci"),
                 ("arith_ops", "extui"),
                 ("torch_ops", "instancenorm")]:
        
        test_cases.append(Case("_".join(name), {
            "mlir_model_file": "static_mlir_model_file",
            "static_mlir_model_file": str(MODELS_DIR / name[0] / (name[1] + ".mlir"))
        }))

    test_cases.append(Case("torch_ops_equal", {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": str(MODELS_DIR / "torch_ops" / "equal.mlir"),
        "input_data": "equal_input_data"
    }))

    test_cases.append(Case("keras_conv_act", {
        "keras_model": "conv_act_model",
        "mlir_model_file": "tflite_mlir_model_file",
        "tflite_model_file": "quantized_tflite_model_file"
    }))

    return test_cases


@pytest.fixture(params=get_test_cases())
def case_config(request):

    if request.param.name == "torch_ops_instancenorm":
        pytest.xfail("not implemented yet")

    return {
        "input_data": "tweaked_random_input_data",
        "torq_compiler_options": ["--torq-disable-slices", "--torq-disable-css"],
        **request.param.data
    }


@pytest.mark.ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):    
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
