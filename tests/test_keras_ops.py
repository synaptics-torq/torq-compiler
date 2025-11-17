import pytest

from .models.keras_models import *
from torq.testing.comparison import compare_test_results
from torq.testing.cases import Case


def get_test_cases():
    test_cases = []

    for params in conv_model_params:
        test_cases.append(Case("conv_model_" + params.idfn(), {
            "keras_model_name": "conv_model",
            "keras_model_params": params
        }))

    for quantize_mode in [False, True]:
        test_cases.append(Case("transpose_conv_model_quantized_" + ("i16" if quantize_mode else "i8"), {
            "keras_model_name": "transpose_conv_model",
            "quantize_to_int16": quantize_mode
        }))

    return test_cases


@pytest.fixture(params=get_test_cases())
def case_config(request):

    return {
        "keras_model": request.param.data['keras_model_name'],
        "keras_model_params": request.param.data.get('keras_model_params', {}),
        "mlir_model_file": "tflite_mlir_model_file",
        "tflite_model_file": "quantized_tflite_model_file",
        "input_data": "tweaked_random_input_data",
        "quantize_to_int16": request.param.data.get("quantize_to_int16", False)
    }


@pytest.mark.ci
def test_keras_model(request, torq_results, tflite_reference_results, case_config):
    compare_test_results(request, torq_results, tflite_reference_results, case_config)
