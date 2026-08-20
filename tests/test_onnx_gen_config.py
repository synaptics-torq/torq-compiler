import json
import pytest

try:
    import iree.compiler
except ImportError:
    pytest.skip("iree package not available", allow_module_level=True)


def _gather_indices_input_range(case):
    """Valid index range for Gather layers with a constant table.

    Discovery feeds random integer inputs drawn from (-40, 40) by default. For
    a Gather whose constant table has fewer rows along the gathered axis (e.g.
    BERT token_type_embeddings with 2 rows) that produces out-of-bounds
    indices, which is undefined behaviour - the compiled host gather reads out
    of bounds and can segfault. When every runtime (non-initializer) input of
    the layer is the indices tensor of a Gather with a constant table,
    restrict the generated indices to [0, table_dim). Returns None when the
    range should not be overridden.
    """
    try:
        model = case.data.model
    except Exception:
        return None

    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    # The table of an extracted Gather layer is typically listed both as an
    # initializer and as a graph input; only non-initializer inputs are
    # actually driven with random data.
    runtime_inputs = {inp.name for inp in graph.input if inp.name not in initializers}
    if not runtime_inputs:
        return None

    table_dim = None
    for node in graph.node:
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        data_name, indices_name = node.input[0], node.input[1]
        # Table must be a constant and indices must be a runtime input.
        if data_name not in initializers or indices_name not in runtime_inputs:
            return None
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
        dims = initializers[data_name].dims
        if not dims:
            return None
        dim = dims[axis % len(dims)]
        table_dim = dim if table_dim is None else min(table_dim, dim)
        runtime_inputs.discard(indices_name)

    if table_dim is None or table_dim <= 0:
        return None

    # Do not clamp if some runtime input is not a gather indices tensor:
    # the single range would apply to that input as well.
    if runtime_inputs:
        return None

    return (0, table_dim)


"""
ONNX Executor Discovery Test Entry Point

cmd:
pytest tests/test_onnx_gen_config.py --model-path=encoder.onnx -v

"""

from torq.gen_config.discovery import pytest_generate_tests

# Fixtures
from torq.gen_config.discovery import (
    reference_results,
    layer_executor_case,
    onnx_layer_model,
    torq_compiler_options,
    comparison_config_for_executor_discovery,
    save_progress,
)

# Core logic imports
from torq.gen_config.discovery import (
    executor_discovery,
    _extract_model_name_from_case,
    _get_subgraph_suffix,
    _maybe_skip_executor,
)


@pytest.fixture
def case_config(request, tmp_path, layer_executor_case, chip_config):
    """Generate case config with executor assignments and tolerances."""

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

    # Keep Gather indices in bounds so random inputs cannot trigger
    # out-of-bounds table accesses (UB / segfault in host-compiled code).
    gather_range = _gather_indices_input_range(case)
    if gather_range is not None:
        base_config["tweaked_input_data_range"] = gather_range

    # Early skip check: skip before expensive fixture setup (compilation)
    model_name = _extract_model_name_from_case(case)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    _maybe_skip_executor(request, layer_id, executor, model_name, subgraph_suffix, layer_executor_case=layer_executor_case)

    # Full model / full subgraph mode: executor assignments provided by fixture.
    # The compiler runs with the default max-producers fuse mode; it falls back
    # to only-patterns per op when the fused producers can't fit in LRAM.
    if executor == "discovered":
        return base_config

    # Layer mode: assign executor to the entire layer
    # Note: Layer tests are for DISCOVERY only. Executor assignment in C++ pass
    # only works for full model tests where line numbers match.
    # The --torq-disable-* flags enforce the executor for layer tests.
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
