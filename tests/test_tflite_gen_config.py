import json
import pytest

try:
    import iree.compiler
except ImportError:
    pytest.skip("iree package not available", allow_module_level=True)

"""
TFLite Executor Discovery Test Entry Point

cmd:
pytest tests/test_tflite_gen_config.py --model-path=model.tflite -v

"""

from torq.gen_config.discovery_tflite import pytest_generate_tests

# Fixtures
from torq.gen_config.discovery_tflite import (
    reference_results,
    layer_executor_case,
    tflite_layer_model,
    gen_config_full_model_input_data,
    torq_compiler_options,
    comparison_config_for_executor_discovery,
    save_progress,
)

# Core logic imports
from torq.gen_config.discovery_tflite import (
    executor_discovery_tflite,
    _extract_model_name_from_case,
    _maybe_skip_executor,
)


@pytest.fixture
def case_config(request, tmp_path, layer_executor_case, chip_config):
    """Generate case config with executor assignments and tolerances."""
    layer_id = layer_executor_case["layer_id"]
    executor = layer_executor_case["executor"]
    case = layer_executor_case["case"]

    # Build base_config FIRST before any potential skip so dependent fixtures
    # still receive valid keys even when this test is skipped.
    base_config = {
        "full_model_input_data": "gen_config_full_model_input_data",
        "tflite_model_file": "tflite_model_path",
        "mlir_model_file": "tflite_mlir_model_file",
        "input_data": "tflite_layer_inputs",
        "comparison_config": "comparison_config_for_executor_discovery",
    }

    # Early skip check: skip before expensive fixture setup (compilation)
    model_name = _extract_model_name_from_case(case)
    _maybe_skip_executor(request, layer_id, executor, model_name)

    # Full model mode: executor assignments provided by fixture
    if executor == "discovered":
        base_config["torq_compiler_options"] = ["--torq-tile-and-fuse-producers-fuse-mode=only-patterns"]
        return base_config

    # Layer mode: assign executor to the entire layer
    json_path = tmp_path / f"torq_gen_config_{executor}.json"
    assignment = {"op_assignments": {layer_id: {"executor": executor}}}
    with open(json_path, "w") as f:
        json.dump(assignment, f, indent=2)

    compiler_options = [f"--torq-executor-map={json_path}"]
    if executor == "nss":
        compiler_options.extend(["--torq-disable-css", "--torq-disable-host"])
    elif executor == "css":
        compiler_options.extend(["--torq-disable-slices", "--torq-disable-host"])
    elif executor == "host":
        compiler_options.extend(["--torq-disable-slices", "--torq-disable-css"])

    compiler_options.append("--torq-tile-and-fuse-producers-fuse-mode=only-patterns")
    base_config["torq_compiler_options"] = compiler_options
    return base_config


def test_executor_discovery(
    request,
    torq_results,
    reference_results,
    case_config,
    layer_executor_case,
    tflite_mlir_model_file,
):
    executor_discovery_tflite(
        request, torq_results, reference_results, case_config,
        layer_executor_case, tflite_mlir_model_file
    )
