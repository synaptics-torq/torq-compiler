"""Keras operator tests: each model is quantized to TFLite (int8 or int16), imported to TOSA and
compiled for Torq; the results are compared with the TFLite runtime.

The models are defined in tests/testdata/keras_ops (see keras_ops_models.py).
"""

import pytest

from .keras_ops_models import (  # noqa: F401 (fixtures)
    keras_ops_case_data,
    keras_ops_cases,
    keras_ops_model,
)
from torq.testing.comparison import compare_test_results


@pytest.fixture
def keras_ops_comparison_config(case_config):
    return case_config["_case"]["config"]["comparison"]


@pytest.fixture(params=keras_ops_cases())
def case_config(request, runtime_hw_type, chip_config):
    data = request.param.data
    config = data["config"]
    cfg = {
        "_case": data,
        "keras_model": "keras_ops_model",
        "keras_model_params": {},
        "mlir_model_file": "tflite_mlir_model_file",
        "tflite_model_file": "quantized_tflite_model_file",
        "input_data": "tweaked_random_input_data",
        "quantize_to_int16": data["quantize_to_int16"],
    }
    if "compiler_options" in config:
        cfg["torq_compiler_options"] = list(config["compiler_options"])
    if "comparison" in config:
        cfg["comparison_config"] = "keras_ops_comparison_config"
    return cfg


def test_keras_model_llvm(request, llvmcpu_reference_results, tflite_reference_results, case_config):
    compare_test_results(request, llvmcpu_reference_results, tflite_reference_results, case_config)


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_keras_model_torq(request, torq_results, tflite_reference_results, case_config):
    compare_test_results(request, torq_results, tflite_reference_results, case_config)
