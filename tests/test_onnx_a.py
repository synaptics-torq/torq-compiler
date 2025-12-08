import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import numpy as np

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_model, onnx_reference_results
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.hf import model_large_512_stream_bf16_onnx, model_large_512_stream_bf16_onnx_name


from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture, versioned_cached_data_fixture

@versioned_unhashable_object_fixture
def get_full_model(request, model_large_512_stream_bf16_onnx):

    model = onnx.load(model_large_512_stream_bf16_onnx)
    # Run shape inference on the original model to get value_info with shapes
    inferred_model = None
    try:
        inferred_model = shape_inference.infer_shapes(model)
    except Exception as e:
        inferred_model = model
    
    return inferred_model


@versioned_cached_data_fixture
def comparison_config_from_dict(request):
    return {
            "fp_avg_tol": 0.03,
            "fp_max_tol": 1
            }


@pytest.fixture
def case_config(request, get_full_model):

    model = "get_full_model"

    return {
        "onnx_model": model,
        "mlir_model_file": "onnx_mlir_model_file",
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_dict"
    }


@pytest.mark.ci
def test_onnx_model_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)