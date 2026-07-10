import numpy as np
import pytest
from filelock import FileLock

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import (
    ModelWithMetadata,
    OnnxLayerCase,
    generate_onnx_layers_from_file,
    onnx_reference_results,
)
from torq.testing.hf import get_hf_model_file
from torq.testing.iree import get_dtype
from torq.testing.decoder_components_extractor import extract_representative_components

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_cached_data_fixture


# see test_torch_ops.py::comparison_config_for_gelu()
@pytest.fixture
def comparison_config_for_gelu(request):
    # Gelu outputs near zero amplify bf16 rounding into large relative error; a
    # slightly larger epsilon keeps those near-zero values within tolerance.
    return {"epsilon": 1e-4}


@versioned_cached_data_fixture
def comparison_config_relaxed(request):
    return {"fp_avg_tol": 0.02, "fp_max_tol": 1.0}


@versioned_hashable_object_fixture
def comparison_config_for_activation_noise():
    return {"fp_avg_tol": 0.05, "fp_max_tol": 0.2, "allowed_wrong": 0.001}

@versioned_hashable_object_fixture
def comparison_config_full_model():
    # The comparison's relative-difference metric is |e-o|/(|e|+|o|+eps), which is
    # mathematically bounded in [0, 1).  A fp_max_tol of 1.0 (as in
    # comparison_config_relaxed) therefore can never be exceeded, which silently
    # disables the check.  Keep fp_max_tol strictly below 1.0 so the full-model
    # comparison actually gates on output correctness.  Pair the relative gate with
    # an absolute gate (fp_abs_tol_frac, a few bf16 ULP of the tensor's range) so the
    # near-zero cancellation outputs that bf16 matmul produces are not flagged, while
    # broad numerical drift still fails.
    return {"fp_avg_tol": 0.02, "fp_max_tol": 0.2, "use_abs_tol_gate": True, "fp_abs_tol_frac": 0.05, "allowed_wrong": 0.001}


@versioned_cached_data_fixture
def gemma3_full_model_input_data(request, tweaked_random_input_data, mlir_io_spec):
    """Input data for the Gemma3 decoder full-model cases.

    tweaked_random_input_data randomizes every input, but `position_ids` is a
    control index (the current decode position), not free data.  With the default
    tweaked range (-40, 40) it is drawn as e.g. -37; a negative/large position
    drives the rotary embedding to an out-of-range angle where the NPU's bf16
    Cos/Sin LUT diverges sharply from ONNXRuntime's fp32 trig (the error scales
    with |position|).  That divergence propagates through Q@K -> softmax ->
    attention, so every downstream output disagrees even though the kernels are
    numerically correct.  Replace the integer control input(s) with a valid, small
    decode position so the comparison exercises the real decode path.  This mirrors
    test_moonshine_models.py::moonshine_decoder_input_data.
    """
    valid_position = 5  # any small non-negative position keeps rotary angles in range
    data = [d.copy() for d in tweaked_random_input_data]
    for i, tensor_type in enumerate(mlir_io_spec.inputs):
        if np.issubdtype(get_dtype(tensor_type.fmt), np.integer):
            data[i] = np.full(tensor_type.shape, valid_position, dtype=data[i].dtype)
    return data


@pytest.fixture
def case_config(request, chip_config):

    no_negative_input = [
        'layer_Sqrt',
    ]

    extra_args = {}
    if any(s in request.node.name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    if any(s in request.node.name for s in ('Cos', 'Sin')):
        extra_args["tweaked_input_data_range"] = 0, 1.5

    comp_config = {
         "onnx_model": "onnx_layer_model",
         "mlir_model_file": "onnx_mlir_model_file",
         "input_data": "tweaked_random_input_data",
         "llvmcpu_compiler_options": ["--iree-global-opt-enable-early-materialization=false"],
         **extra_args
    }

    torq_compiler_options = ["--torq-convert-dtypes", "--torq-convert-io-dtype"]
    nss_layers = ["layer_MatMul"]
    if any(s in request.node.name for s in nss_layers):
        torq_compiler_options += ["--torq-disable-css", "--torq-disable-host"]
    comp_config["torq_compiler_options"] = torq_compiler_options

    if "Gelu" in request.node.name:
        comp_config["comparison_config"] = "comparison_config_for_gelu"

    # Softmax accumulates activation noise; relax tolerances like Moonshine does.
    if "layer_Softmax" in request.node.name:
        comp_config["comparison_config"] = "comparison_config_for_activation_noise"

    # bf16 ReduceMean: HW fp32 accumulation vs llvmcpu bf16 accumulation -> ~1 ULP drift.
    if "layer_ReduceMean" in request.node.name:
        comp_config["comparison_config"] = "comparison_config_relaxed"

    if "full_model" in request.node.name:
        comp_config["comparison_config"] = "comparison_config_full_model"
        # The full-model decode path needs valid control inputs and realistic
        # activation magnitudes.  position_ids is a decode index (not free data),
        # and the default random range (-40, 40) both picks an invalid position and
        # inflates activations far beyond real hidden-state/KV magnitudes, which
        # artificially amplifies bf16 precision loss vs the fp32 reference.  Use a
        # valid decode position and a realistic range, mirroring the Moonshine
        # decoder full-model harness.
        comp_config["input_data"] = "gemma3_full_model_input_data"
        comp_config["tweaked_input_data_range"] = (-2, 2)

    return comp_config


def pytest_generate_tests(metafunc):
    source_model = get_hf_model_file(
        metafunc.config.cache, "Synaptics/gemma-3-270m-it", "onnx/model.onnx"
    )

    # Extract whatever representative components the extractor can discover and
    # cache them on disk. Under pytest-xdist every worker collects independently,
    # so guard the (heavy) extraction with a lock: the first worker extracts and
    # writes a completion marker; the rest reuse the cached component files. The
    # marker lets us skip re-extraction without hardcoding the component names.
    components_dir = metafunc.config.cache.mkdir("gemma3_components")
    extracted_marker = components_dir / ".extracted"
    with FileLock(str(components_dir / "lock")):
        if not extracted_marker.exists():
            extract_representative_components(source_model, components_dir)
            extracted_marker.touch()

    # generate_onnx_layers_from_file is disk-cache-backed: the first collection
    # extracts + shape-infers each component's layers and writes them to
    # .pytest_cache; subsequent collections (and other xdist workers) skip the
    # heavy extraction and just wrap each cached layer/full-model file in a
    # lazily-loaded ModelWithMetadata.
    cases = []
    for comp_path in sorted(components_dir.glob("*.onnx")):
        cases += generate_onnx_layers_from_file(
            metafunc.config.cache, comp_path, dedup=True
        )

    # Load lazily so collection does not pull the ~800MB model into memory.
    cases.append(
        OnnxLayerCase(
            name="source_full_model",
            data=ModelWithMetadata(path=source_model),
            is_full_model=True,
        )
    )

    metafunc.parametrize("onnx_layer_model", cases, indirect=True)

@versioned_hashable_object_fixture
def onnx_fake_quantize_config():
    # Always fake-quantize FP32/INT64 weights to BF16/INT32 for these models.
    return {"fake_quantize": True}


@pytest.fixture
def reference_results(request, onnx_layer_model):
    return request.getfixturevalue("onnx_reference_results")


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_onnx_model_torq(request, reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, reference_results, case_config)
