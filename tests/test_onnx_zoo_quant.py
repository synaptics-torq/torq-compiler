import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_hf, composite_reference_results
from torq.testing.versioned_fixtures import versioned_hashable_object_fixture

"""ONNX Model Zoo quantization regression test.

This module downloads models from the Hugging Face ``onnxmodelzoo`` organization,
quantizes each full model with the TORQ ONNX quantizer, compiles it for the TORQ
backend, and compares against a reference execution.  It is intended as a
regression check for the quantization path across publicly available ONNX Model
Zoo models.

Quantization is enabled by overriding the shared ``onnx_quant_config`` fixture
locally in this module.  This avoids mutating the global pytest config object
and keeps the setting scoped to tests in this file.
"""


# Hugging Face ONNX Model Zoo models to regression-test with quantization.
# Each entry is (repo_id, filename).  Add more tuples here to extend coverage.
_MODELS = [
    ("onnxmodelzoo/resnet18_Opset18_timm", "resnet18_Opset18_timm.onnx"),
]


@versioned_hashable_object_fixture
def comparison_config_onnx_zoo_quant():
    """Relaxed integer-comparison tolerance for the quantized full model.

    The current values are intentionally loose: the quantizer in this test
    uses random calibration data, so exact bit-wise agreement with the
    reference is not expected.  Real model accuracy will be validated once
    proper calibration data is added.
    """
    return {"int_tol": 5, "int_thld": 10}


@versioned_hashable_object_fixture
def onnx_quant_config():
    """Force the shared ONNX fixtures to use quantization for this module.

    Overrides the global default so that running this file alone always
    exercises the quantized path, without leaking the setting to other test
    modules collected in the same pytest process.
    """
    return {
        "quantize": True,
        "per_channel": False,
        "full_integer": True,
        "quant_format": "qdq",
        "quant_dtype": "A8W8",
    }


@pytest.fixture
def case_config(request, chip_config):
    """Standard full-model config for the quantized ONNX Model Zoo model."""

    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip:
        pytest.xfail("AssertionError: Nans differ")

    return {
        "onnx_model": "onnx_layer_model",
        "mlir_model_file": "onnx_mlir_model_file",
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_onnx_zoo_quant",
    }


def pytest_generate_tests(metafunc):
    """Generate one full-model test case per entry in _MODELS."""

    config = metafunc.config

    full_model_cases = []
    for repo_id, filename in _MODELS:
        all_cases = generate_onnx_layers_from_hf(
            config.cache,
            repo_id,
            filename,
            node_groups=None,
            dedup=True,
        )
        model_full_cases = [c for c in all_cases if c.is_full_model]
        assert model_full_cases, f"no full-model case generated for {repo_id}/{filename}"
        full_model_cases.extend(model_full_cases)

    metafunc.parametrize(
        "onnx_layer_model",
        full_model_cases,
        indirect=True,
        ids=[c.name for c in full_model_cases],
    )


@pytest.mark.ci
def test_onnx_zoo_quant_llvmcpu_torq(
    request, composite_reference_results, torq_results, case_config, onnx_layer_model
):
    compare_test_results(
        request, torq_results, composite_reference_results, case_config
    )
