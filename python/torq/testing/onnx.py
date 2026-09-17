import copy as _copy
from functools import wraps
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import onnxruntime
import pytest

from torq.lab.model_tools.dtype_conversion.onnx import convert_model

from torq.testing.cases import Case
from torq.testing.hf import get_hf_model_file

from .versioned_fixtures import (
    versioned_cached_data_fixture,
    versioned_generated_file_fixture,
    versioned_hashable_object_fixture,
    versioned_unhashable_object_fixture,
    VersionedUncachedData,
)
from torq.lab.quantization.onnx.fake import onnx_fake_quantize
from .quantization import onnx_fake_quantize_config

from torq.testing.quantize_onnx import (
    add_onnx_static_quantization_options,
    is_model_quantized,
    onnx_static_quantize,
)

# Re-export the helpers from torq.lab.model_tools.extraction.onnx so the
# torq.testing.onnx import surface keeps working.
from torq.lab.model_tools.extraction.onnx.layers import (
    ModelWithMetadata,
    OnnxLayerCase,
    _format_layer_node_name,
    _load_cached_layers,
    generate_onnx_layers_from_file,
    generate_onnx_layers_from_model,
    get_full_model,
    is_onnx_qdq_wrapper_layer,
    model_signature,
)
from torq.lab.model_tools.extraction.onnx.subgraphs import extract_onnx_subgraph
from torq.lab.model_tools.importers.onnx import convert_onnx_to_mlir

"""
Fixtures and utilities for testing ONNX models.

The layer/subgraph extraction, model-signature, layer-cache, and ONNX->MLIR
import helpers live in ``torq.lab.model_tools.extraction.onnx`` and are re-exported above;
this module adds the hooks and fixtures.
"""

_pytest_config = None


def pytest_addoption(parser):
    parser.addoption(
        "--onnx-print-layer-info",
        action="store_true",
        default=False,
        help="Print original ONNX node names for generated ONNX layer tests during setup",
    )
    # Shared ONNX quantization flags (also used by torq-gen-config).
    add_onnx_static_quantization_options(parser)


def generate_onnx_layers_from_hf(cache, repo_id, filename, node_groups=None, dedup=True):
    model_file = get_hf_model_file(cache, repo_id, filename)
    return _load_cached_layers(
        cache, model_file, Path(filename).stem, node_groups, dedup)


# ---- Pytest fixtures ----

@pytest.fixture
def onnx_model(request, case_config):
    return request.getfixturevalue(case_config['onnx_model'])


def onnx_model_fixture(fun):
    """Decorator for fixtures that build an in-memory onnx.ModelProto (mirror of
    tensorflow.keras_model_fixture). The wrapped fixture is consumed via the
    existing onnx_model -> onnx_model_file -> onnx_mlir_model_file chain, so a
    parametric builder needs no new downstream plumbing."""
    @versioned_unhashable_object_fixture
    @wraps(fun)
    def wrapper(**kwargs):
        request = kwargs.get("request")
        if request is None:
            raise ValueError("onnx_model_fixture requires a request fixture")
        record_property = request.getfixturevalue("record_property")
        record_property("compiler_input", f"onnx:{fun.__module__}.{fun.__qualname__}")
        return fun(**kwargs)

    return wrapper


@versioned_generated_file_fixture("onnx")
def onnx_model_file(request, versioned_file, onnx_model, onnx_fake_quantize_config):
    if onnx_fake_quantize_config["fake_quantize"]:
        onnx_model = onnx_fake_quantize(_copy.deepcopy(onnx_model))
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, versioned_file)


@pytest.fixture
def onnx_source_model_file(request, onnx_model):
    """Return the source ONNX model path used to expand per-layer calibration data.

    For generated layer tests the source model path is carried on the layer
    case metadata. For full-model or hand-written fixtures it is typically
    absent, in which case this fixture returns ``None``.
    """
    model = onnx_model
    if isinstance(model, VersionedUncachedData):
        model = model.data
    source_path = getattr(model, "source_model_path", None)
    if source_path:
        return Path(source_path)
    return None


@versioned_hashable_object_fixture
def onnx_bf16_config(request):
    """Return BF16 conversion config for version hashing.

    This ensures cache invalidation when --auto-convert-bf16 flag changes.
    """
    return {
        "auto_convert_bf16": request.config.getoption("--auto-convert-bf16", default=False)
    }


@versioned_generated_file_fixture("onnx_bf16")
def onnx_bf16_model_file(request, versioned_file, onnx_model_file, onnx_bf16_config):
    """Convert FP32 ONNX to BF16 if --auto-convert-bf16 is enabled.

    Always saves to versioned_file location (the decorator always returns versioned_file).

    Note: onnx_model_file is a Path (versioned_generated_file_fixture unwraps VersionedFile).
    """
    import shutil
    use_bf16 = request.config.getoption("--auto-convert-bf16", default=False)

    if not use_bf16:
        # No conversion - copy original model to versioned location
        print(f"[BF16] Conversion disabled, copying original model to {versioned_file}")
        shutil.copy(str(onnx_model_file), str(versioned_file))
        return versioned_file

    print(f"[BF16] Converting {onnx_model_file.name} to BF16...")
    convert_model(onnx_model_file, versioned_file, "bf16", convert_io=True)
    print(f"[BF16] Saved to: {versioned_file}")

    return versioned_file


@versioned_hashable_object_fixture
def onnx_int32_config(request):
    """Return INT32 conversion config for version hashing.

    This ensures cache invalidation when --auto-convert-int32 flag changes.
    """
    return {
        "auto_convert_int32": request.config.getoption("--auto-convert-int32", default=False)
    }


@versioned_generated_file_fixture("onnx_int32")
def onnx_int32_model_file(request, versioned_file, onnx_bf16_model_file, onnx_int32_config):
    """Convert INT64 tensors in an ONNX model to INT32 if --auto-convert-int32 is enabled.

    Chained after the BF16 step so that when both conversions are enabled the
    order is bf16 first, then int32. Always saves to versioned_file location
    (the decorator always returns versioned_file).

    Note: onnx_bf16_model_file is a Path (versioned_generated_file_fixture
    unwraps VersionedFile).
    """
    import shutil
    use_int32 = onnx_int32_config["auto_convert_int32"]

    if not use_int32:
        # No conversion - pass the (possibly BF16-converted) model through
        print(f"[INT32] Conversion disabled, copying model to {versioned_file}")
        shutil.copy(str(onnx_bf16_model_file), str(versioned_file))
        return versioned_file

    print(f"[INT32] Converting INT64 tensors in {onnx_bf16_model_file.name} to INT32...")
    convert_model(onnx_bf16_model_file, versioned_file, "int32", convert_io=True)
    print(f"[INT32] Saved to: {versioned_file}")

    return versioned_file


@versioned_hashable_object_fixture
def onnx_quant_config(request):
    """Return quantization config for version hashing.

    This ensures cache invalidation when --quantize or its sub-options change.
    Test modules can override this fixture locally to force quantization without
    mutating the global pytest config object.
    """
    return {
        "quantize": request.config.getoption("--quantize", default=False),
        "per_channel": request.config.getoption("--per-channel", default=False),
        "full_integer": request.config.getoption("--full-integer", default=False),
        "quant_format": request.config.getoption("--quant-format", default="qdq"),
        "quant_dtype": request.config.getoption("--quant-dtype", default="A8W8"),
    }


@versioned_generated_file_fixture("onnx_quantized")
def onnx_quantized_model_file(
    request, versioned_file, onnx_model_file, onnx_quant_config
):
    """Quantize ONNX model to int8 if --quantize is enabled.

    Uses the original ONNX model as the source. If quantization is disabled,
    the original model is copied to the versioned location.

    The actual settings are taken from the ``onnx_quant_config`` fixture, so
    individual test modules can override them without touching the global
    pytest config.

    Note: onnx_model_file is a Path object
    (versioned_generated_file_fixture unwraps VersionedFile to Path).
    """
    import shutil

    use_quantize = onnx_quant_config["quantize"]
    per_channel = onnx_quant_config["per_channel"]
    full_integer = onnx_quant_config["full_integer"]
    quant_format = onnx_quant_config["quant_format"]
    quant_dtype = onnx_quant_config["quant_dtype"]

    source_path = onnx_model_file

    if not use_quantize:
        print(f"[Quantize] Quantization disabled, copying source model to {versioned_file}")
        shutil.copy(str(source_path), str(versioned_file))
        return versioned_file

    model = onnx.load(str(source_path))
    if is_model_quantized(model):
        print(f"[Quantize] Model already quantized, copying to {versioned_file}")
        shutil.copy(str(source_path), str(versioned_file))
        return versioned_file

    print(
        f"[Quantize] Quantizing {source_path.name} "
        f"(quant_format={quant_format}, quant_dtype={quant_dtype}, "
        f"per_channel={per_channel}, full_integer={full_integer})..."
    )
    quantized_model = onnx_static_quantize(
        model,
        per_channel=per_channel,
        full_integer=full_integer,
        quant_format=quant_format,
        quant_dtype=quant_dtype,
    )
    onnx.save(quantized_model, str(versioned_file))
    print(f"[Quantize] Saved to: {versioned_file}")
    return versioned_file


@versioned_generated_file_fixture("mlir")
def onnx_mlir_model_file(request, versioned_file, onnx_model_file, onnx_bf16_model_file, onnx_bf16_config, onnx_int32_model_file, onnx_int32_config, onnx_quantized_model_file, onnx_quant_config):
    """Convert ONNX model to MLIR with enhanced error diagnostics.

    Uses quantized model if enabled by ``onnx_quant_config``, otherwise the
    INT32 model if --auto-convert-int32 is enabled (it chains the BF16 step,
    so bf16 conversion is applied first when both flags are set), otherwise
    the BF16 model if --auto-convert-bf16 is enabled, otherwise the original
    model. This ensures the compiler receives the correctly converted model
    based on user options.

    Note: onnx_model_file, onnx_bf16_model_file, onnx_int32_model_file, and
    onnx_quantized_model_file are Path objects (versioned_generated_file_fixture
    unwraps VersionedFile to Path).
    """
    use_quantize = onnx_quant_config["quantize"]
    use_bf16 = request.config.getoption("--auto-convert-bf16", default=False)
    use_int32 = onnx_int32_config["auto_convert_int32"]

    if use_quantize:
        model_path = onnx_quantized_model_file
        print(f"[Quantize] Using quantized model for MLIR conversion: {model_path}")
    elif use_int32:
        model_path = onnx_int32_model_file
        print(f"[INT32] Using INT32 model for MLIR conversion: {model_path}")
    elif use_bf16:
        model_path = onnx_bf16_model_file
        print(f"[BF16] Using BF16 model for MLIR conversion: {model_path}")
    else:
        model_path = onnx_model_file

    convert_onnx_to_mlir(model_path, versioned_file)

@versioned_hashable_object_fixture
def onnx_ref_data_cache_key():
    return "round-trip-converted-inputs-v1"

@versioned_cached_data_fixture
def onnx_ref_data(request, input_data, convert_io_dtypes_policy, onnx_ref_data_cache_key):
    return convert_io_dtypes_policy.round_trip_inputs(input_data)

def _prepare_onnxruntime_feed_value(data):
    """Convert reference feed arrays that ONNX Runtime's Python API cannot accept."""
    if np.dtype(data.dtype) == np.dtype(ml_dtypes.bfloat16):
        return data.astype(np.float32)
    return data

@versioned_hashable_object_fixture
def onnx_params(case_config):
    return {"opset": case_config.get("opset", 20)}

@versioned_unhashable_object_fixture
def onnx_reference_results(request, onnx_ref_data, convert_io_dtypes_policy):
    onnx_model_file = request.getfixturevalue("onnx_model_file").file_path
    onnx_model = onnx.load(str(onnx_model_file))
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
    ort_inputs = {
        inp.name: _prepare_onnxruntime_feed_value(onnx_ref_data[i])
        for i, inp in enumerate(ort_session.get_inputs())
    }
    outputs = ort_session.run(None, ort_inputs)
    return convert_io_dtypes_policy.round_trip_outputs(outputs)


@pytest.fixture
def onnx_layer_model(request):
    case_name = request.param.name
    model_data = request.param.data
    version = "onnx_layer_model_" + case_name

    if isinstance(request.param, OnnxLayerCase):
        if request.config.getoption("--onnx-print-layer-info") and not request.param.is_full_model:
            print(_format_layer_node_name(request.param))

    record_property = request.getfixturevalue("record_property")
    record_property("compiler_input", f"onnx:{case_name}")

    return VersionedUncachedData(data=model_data, version=version)


# Re-export numpy executor functions and fixture for backward compatibility
# These are defined in .numpy to avoid circular dependencies with .torch
from .numpy import (
    has_bf16_matmul,
    has_bf16_einsum,
    has_gelu,
    numpy_maxpool,
    execute_onnx_model_numpy,
    numpy_gelu_reference_results,
    numpy_reference_results,
)


@versioned_unhashable_object_fixture
def composite_reference_results(request, input_data, onnx_quant_config):
    """
    Generate reference using a chained fallback strategy:
    1. ONNXRuntime (fastest, most accurate for f32)
    2. numpy fallback (for bf16 MatMul/Einsum/MaxPool)
    3. llvmcpu fallback (IREE reference compilation)
    4. torch fallback (last resort for bf16 models with unsupported ops)

    When quantization is enabled (via --quantize or by overriding the
    ``onnx_quant_config`` fixture), the ONNXRuntime reference runs the quantized
    ONNX model (via onnx_quantized_model_file) so the TORQ compiled output is
    compared against the same quantized integer graph instead of the original
    FP32 model.
    """
    # Try ONNX-based paths first if an ONNX model is available.
    try:
        use_quantize = onnx_quant_config["quantize"]
        if use_quantize:
            onnx_model_file = request.getfixturevalue("onnx_quantized_model_file").file_path
        else:
            onnx_model_file = request.getfixturevalue("onnx_model_file").file_path
        onnx_model = onnx.load(str(onnx_model_file))

        # 1. Try ONNXRuntime first
        if not has_bf16_matmul(onnx_model) or not has_bf16_einsum(onnx_model):
            try:
                ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
                ort_inputs = {inp.name: input_data[i] for i, inp in enumerate(ort_session.get_inputs())}
                return ort_session.run(None, ort_inputs)
            except Exception:
                pass

        # 2. Try numpy fallback
        try:
            return execute_onnx_model_numpy(onnx_model, input_data)
        except Exception:
            pass
    except Exception:
        pass

    # 3. Try llvmcpu fallback
    try:
        return request.getfixturevalue("llvmcpu_reference_results").data
    except Exception:
        pass

    # 4. Last resort: torch fallback
    print("Warning: All previous references (ONNXRuntime, numpy, llvmcpu) failed, falling back to torch reference")
    return request.getfixturevalue("torch_reference_results").data
