import json
import pytest

# disabling for the moment as it makes xdist workers to die at the moment
import os
if not os.environ.get("TORQ_ENABLE_EXECUTOR_DISCOVERY_PLUGIN"):
    pytest.skip("Test not enabled with TORQ_ENABLE_EXECUTOR_DISCOVERY_PLUGIN environment variable set", allow_module_level=True)

"""
ONNX Executor Discovery Test Entry Point

cmd:
pytest tests/test_onnx_executor_discovery.py --model-path=encoder.onnx -v

"""

from torq.executor_discovery.executor_discovery_onnx import pytest_generate_tests

# Fixtures
from torq.executor_discovery.executor_discovery_onnx import (
    reference_results,
    layer_executor_case,
    onnx_layer_model,
    torq_compiler_options,
    comparison_config_for_executor_discovery,
    save_progress,
)

# Core logic imports
from torq.executor_discovery.executor_discovery_onnx import (
    executor_discovery,
    _extract_model_name_from_case,
    _get_subgraph_suffix,
    _maybe_skip_executor,
)


@pytest.fixture
def case_config(request, tmp_path, layer_executor_case, chip_config):
    """Generate case config with executor assignments and tolerances."""

    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip:
        pytest.xfail("AssertionError: Nans differ")

    layer_id = layer_executor_case["layer_id"]
    executor = layer_executor_case["executor"]
    is_subgraph = layer_executor_case.get("is_subgraph", False)
    case = layer_executor_case["case"]

    # Build base_config FIRST before any potential skip
    # This ensures other fixtures that depend on case_config keys get valid values
    # even when this test is skipped
    base_config = {
        "onnx_model": "onnx_layer_model",
        "mlir_model_file": "onnx_mlir_model_file",
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_for_executor_discovery",
    }

    # Early skip check: skip before expensive fixture setup (compilation)
    model_name = _extract_model_name_from_case(case)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    _maybe_skip_executor(request, layer_id, executor, model_name, subgraph_suffix)

    # Full model / full subgraph mode: executor assignments provided by fixture
    if executor == "discovered":
        base_config["torq_compiler_options"] = ["--torq-enable-torq-hl-tiling"]
        return base_config

    # Layer mode: assign executor to the entire layer
    # Note: Layer tests are for DISCOVERY only. Executor assignment in C++ pass
    # only works for full model tests where line numbers match.
    # The --torq-disable-* flags enforce the executor for layer tests.
    json_path = tmp_path / f"executor_assignments_{executor}.json"
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

    compiler_options.append("--torq-enable-torq-hl-tiling")
    base_config["torq_compiler_options"] = compiler_options
    return base_config


def test_executor_discovery(
    request,
    torq_results,
    reference_results,
    case_config,
    layer_executor_case,
    onnx_mlir_model_file,
):
    executor_discovery(
        request, torq_results, reference_results, case_config,
        layer_executor_case, onnx_mlir_model_file
    )
