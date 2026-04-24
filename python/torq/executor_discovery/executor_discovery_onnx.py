import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import onnx
import pytest

from torq.testing.iree import list_files
from torq.testing.onnx import (
    generate_onnx_layers_from_file,
    generate_onnx_layers_from_model,
    extract_onnx_subgraph,
    model_signature,
)
from torq.testing.cases import Case
from torq.testing.versioned_fixtures import (
    VersionedUncachedData,
    versioned_hashable_object_fixture,
)

from torq.executor_discovery._utils import (
    build_report_lines,
    extract_line_numbers_from_mlir,
    format_per_layer_status_table,
    parse_diff_metrics,
)

"""
ONNX Executor Discovery Test

Discovers the appropriate executor (NSS/CSS/Host) for each layer of an ONNX model.
Tests each layer on NSS first, then CSS, then Host, recording which one works.

Executor Discovery and Assignment Flow:

    # Step 1: Run layer discovery tests to find optimal executor for each layer
    # This generates executor_assignments_<model>.json with discovery results
    pytest tests/test_onnx_executor_discovery.py -v -k "encoder_small_layer_" \
        --model-path=./tests/testdata/onnx_models/encoder_small.onnx --recompute-cache

    # Step 2: Run full model test with discovered executor assignments
    # The torq_executor_assignments_json fixture provides the executor assignments
    # C++ ExecutorAssignmentPass uses line:column matching to assign executors
    pytest tests/test_onnx_executor_discovery.py \
        --model-path=./tests/testdata/onnx_models/encoder_small.onnx \
        -v -k "full_model" --debug-ir=tmp --recompute-cache


Design Details:

    Step 1: Layer Discovery Tests
    For each layer (e.g., Tanh, Reshape), test NSS -> CSS -> Host in priority order.
    Save results to executor_assignments_<model>.json with three status types:
    - "success": Test passed
    - "difference": Accuracy failure (numerical difference)

    Users can re-run specific tests with updated tolerance values:
    pytest tests/test_onnx_executor_discovery.py -v -k "encoder_small_layer_Tanh_0_nss" \
        --model-path=./tests/testdata/onnx_models/encoder_small.onnx -v --recompute-cache

    Step 2: Full Model Test
    1. Generate full model MLIR from ONNX
    2. Extract CORRECT line numbers from full model MLIR:
       - Tanh at line 10 -> "10:10"
       - Reshape at line 11 -> "11:10"
    3. Update JSON with correct line numbers (replaces temporary "Tanh" with "10:10")
    4. torq_executor_assignments_json fixture provides executor assignments JSON
       (discovery format with 'ops' key, or compiler format with 'op_assignments' key)
    5. C++ ExecutorAssignmentPass:
       - Loads executor assignments from JSON
       - For each operation, extracts line:column from CallSiteLoc
       - Matches line:column with assignment keys
       - Sets torq-executor attribute

    Note: Line numbers from layer MLIRs (4:10, 6:10) are different from full model
    MLIR (10:10, 11:10). The fixture extracts correct line numbers directly from
    full model MLIR to ensure proper matching.

JSON Format (executor_assignments_<model>.json):
{
    "version": "1.1",
    "model_name": "encoder",
    "default_tolerance": {"fp_avg_tol": 0.01, "fp_max_tol": 0.01},
    "ops": {
        "Tanh_/Tanh_output_0": {
            "executors": {
                "nss": {
                    "status": "success",
                    "timing": {"runtime_ms": 15.2, "total_ms": 150.5}
                },
                "css": {"status": "difference", ...},
                "host": {"status": "success", ...}
            },
            "expected_executor": "nss",
            "mlir_location": "10:10"
        }
    }
}

Compiler JSON Formats (C++ ExecutorAssignmentPass accepts both):
- Discovery format: {"ops": {"Conv_0": {"recommended_executor": "nss", "mlir_location": "10:10"}}}
- Compiler format: {"op_assignments": {"10:10": {"executor": "nss"}}}
"""

DEFAULT_TOLERANCE = {"fp_avg_tol": 0.01, "fp_max_tol": 0.01}
EXECUTOR_ORDER = ["nss", "css", "host"]
TIMING_PRECISION = 3  # Decimal places for timing values

def _verify_import_ordering(
    onnx_nodes: list, mlir_ops: List[Tuple[str, str]]
) -> Tuple[bool, List[str]]:
    """Verify that MLIR ops match ONNX nodes in order.

    Returns: (is_valid, list_of_warnings)
    """
    warnings = []

    # Filter Constant nodes from ONNX
    onnx_ops = [(n.op_type, n.name) for n in onnx_nodes if n.op_type != "Constant"]

    # Check counts match
    if len(onnx_ops) != len(mlir_ops):
        warnings.append(
            f"COUNT MISMATCH: ONNX has {len(onnx_ops)} non-Constant ops, "
            f"MLIR has {len(mlir_ops)} torch.operator ops"
        )
        return False, warnings

    # Check op types match at each position
    type_mismatches = []
    for i, ((onnx_type, onnx_name), (mlir_type, _)) in enumerate(
        zip(onnx_ops, mlir_ops)
    ):
        if onnx_type != mlir_type:
            type_mismatches.append(
                f"  Position {i}: ONNX={onnx_type} (name='{onnx_name}'), MLIR={mlir_type}"
            )

    if type_mismatches:
        warnings.append(f"OP TYPE MISMATCHES ({len(type_mismatches)}):")
        warnings.extend(type_mismatches[:5])  # Show first 5
        if len(type_mismatches) > 5:
            warnings.append(f"  ... and {len(type_mismatches) - 5} more")
        return False, warnings

    return True, warnings


def _build_onnx_to_mlir_mapping(onnx_model_path: Path, mlir_file: Path) -> Dict[str, str]:
    """Build mapping from ONNX node names to full MLIR line numbers.

    Returns: {onnx_node_name -> "line:column"}
    """
    import onnx
    import subprocess

    try:
        # Generate full model MLIR from ONNX using iree-import-onnx
        mlir_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "iree.compiler.tools.import_onnx",
             str(onnx_model_path), "-o", str(mlir_file), "--data-prop"],
            check=True,
            capture_output=True,
        )

        # Extract all operations from full MLIR
        all_ops = extract_line_numbers_from_mlir(mlir_file)

        # Load ONNX model to get node names
        model = onnx.load(str(onnx_model_path))

        # Verify ordering assumptions
        is_valid, warnings = _verify_import_ordering(model.graph.node, all_ops)

        if warnings:
            _discovery_log("[ONNXtoMLIR] Verification:")
            for w in warnings:
                _discovery_log(f"  ! {w}")

        if not is_valid:
            _discovery_log(
                "[ONNXtoMLIR] WARNING: Import ordering verification failed. "
                "Mapping may be incorrect. See test_onnx_import_order.py"
            )

        # Build mapping: onnx_node_name -> line:column
        # Using position-based matching (most reliable)
        mapping = {}
        op_idx = 0
        for node in model.graph.node:
            if node.op_type == "Constant":
                continue
            if op_idx < len(all_ops):
                op_type, line_col = all_ops[op_idx]
                # Use output[0] as the key to match layer_id format
                if node.output:
                    node_name = f"{node.op_type}_{node.output[0]}"
                    mapping[node_name] = line_col
                op_idx += 1

        onnx_count = len([n for n in model.graph.node if n.op_type != "Constant"])
        _discovery_log(
            f"[ONNXtoMLIR] Mapping: {len(mapping)}/{onnx_count} ops, "
            f"valid={is_valid}"
        )

        return mapping
    except Exception as e:
        _discovery_log(f"[ONNXtoMLIR] Error building mapping: {e}")
        return {}


class ExecutorDiscoveryState:
    """Global state for executor discovery results."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, Dict]] = {}  # layer_id -> executor -> result
        self.locations: Dict[str, str] = {}  # layer_id -> mlir_location (op_type for layer tests)
        self.node_indices: Dict[str, int] = {}  # layer_id -> node_index in ONNX graph
        self.mlir_files: Dict[str, Path] = {}  # layer_id -> mlir file path
        # layer_id -> line:column from full model MLIR
        self.full_mlir_locations: Dict[str, str] = {}
        # Full model comparison metrics (populated when running full model test)
        self.full_model_metrics: Optional[Dict[str, Any]] = None

    def record_result(
        self,
        layer_id: str,
        executor: str,
        status: str,
        tolerance_used: Dict[str, float],
        max_diff: Optional[Dict] = None,
        failure_report: Optional[Dict] = None,
        timing: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record test result for a layer/executor."""
        if layer_id not in self.results:
            self.results[layer_id] = {}

        result: Dict[str, Any] = {"status": status, "tolerance_used": tolerance_used}
        if max_diff:
            result["max_diff"] = max_diff
        if failure_report:
            result["failure_report"] = failure_report
        if timing:
            result["timing"] = timing

        self.results[layer_id][executor] = result

    def record_metadata(
        self,
        layer_id: str,
        node_index: Optional[int] = None,
        mlir_location: Optional[str] = None,
        full_mlir_location: Optional[str] = None,
        mlir_file: Optional[Path] = None,
    ) -> None:
        """Record metadata for a layer."""
        if node_index is not None:
            self.node_indices[layer_id] = node_index
        if mlir_location:
            self.locations[layer_id] = mlir_location
        if full_mlir_location:
            self.full_mlir_locations[layer_id] = full_mlir_location
        if mlir_file:
            self.mlir_files[layer_id] = mlir_file

    def load_from_json(self, json_data: Dict[str, Any]) -> None:
        """Load existing results from JSON data (for skip mode consistency)."""
        ops = json_data.get("ops", {})
        loaded_count = 0
        for layer_id, op_data in ops.items():
            executors = op_data.get("executors", {})
            for executor, result in executors.items():
                # Only load if not already recorded
                if layer_id not in self.results:
                    self.results[layer_id] = {}
                if executor not in self.results[layer_id]:
                    self.results[layer_id][executor] = result
                    loaded_count += 1

            # Load metadata
            node_index = op_data.get("_node_index")
            if node_index is not None and layer_id not in self.node_indices:
                self.node_indices[layer_id] = node_index
            mlir_loc = op_data.get("mlir_location")
            if mlir_loc and layer_id not in self.locations:
                self.locations[layer_id] = mlir_loc

        if loaded_count > 0:
            _discovery_log(f"Loaded {loaded_count} cached results from existing JSON")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        status_counts = {"success": 0, "difference": 0, "error": 0}
        executor_counts = {executor: 0 for executor in EXECUTOR_ORDER}

        # Timing statistics
        timing_available = False
        executor_timing = {executor: [] for executor in EXECUTOR_ORDER}

        for executors in self.results.values():
            for executor, result in executors.items():
                status = result.get("status", "error")
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "success" and executor in executor_counts:
                    executor_counts[executor] += 1

                # Collect timing data
                timing = result.get("timing")
                if timing and "runtime_ms" in timing:
                    timing_available = True
                    executor_timing[executor].append(timing["runtime_ms"])

        summary = {
            "total_layers": len(self.results),
            "status_counts": status_counts,
            "executor_counts": executor_counts,
        }

        # Add timing summary if available
        if timing_available:
            timing_summary = {}
            for executor in EXECUTOR_ORDER:
                times = executor_timing[executor]
                if times:
                    timing_summary[executor] = {
                        "avg_ms": round(sum(times) / len(times), TIMING_PRECISION),
                        "min_ms": round(min(times), TIMING_PRECISION),
                        "max_ms": round(max(times), TIMING_PRECISION),
                        "samples": len(times),
                    }
            if timing_summary:
                summary["timing_summary"] = timing_summary

        return summary


# Global state
_discovery_state = ExecutorDiscoveryState()

def _discovery_log(msg: str) -> None:
    """Write discovery diagnostic output to stderr (redirected to log file when configured)."""
    print(msg, file=sys.stderr)


def _extract_op_type_from_layer(mlir_file: Path, target_op_type: str) -> Optional[str]:
    """Extract operation type from layer MLIR file."""
    if not mlir_file.exists():
        return None
    try:
        content = mlir_file.read_text()
        pattern = rf'torch\.operator\s+"onnx\.({target_op_type})"'
        if re.search(pattern, content, re.IGNORECASE):
            return target_op_type
    except Exception as e:
        _discovery_log(f"[LocationExtract] Error reading {mlir_file}: {e}")
    return None


def _is_line_column_format(location: str) -> bool:
    """Check if location is in 'line:column' format (e.g., '10:10')."""
    return bool(location and re.match(r"^\d+:\d+$", location))


def _get_skipped_executors(config) -> set:
    """Get set of executors to skip from --skip-executors option.

    Args:
        config: pytest config object or Config object with getoption method

    Returns:
        Set of executor names to skip (e.g., {'nss', 'css'})
    """
    skip_executors_option = config.getoption("--skip-executors", default=None)
    if skip_executors_option:
        return {e.strip().lower() for e in skip_executors_option.split(",")}
    return set()


def _get_layer_id_from_case(case) -> str:
    """Extract layer_id (opType_outputName) from a Case's ONNX model."""
    model_wrapper = case.data
    model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    graph = model.graph
    return (
        f"{graph.node[0].op_type}_{graph.node[0].output[0]}"
        if graph.node
        else "unknown_layer"
    )


def _build_duplicate_layer_map(cases) -> Dict[str, str]:
    """Build a mapping from duplicate layer_id to source layer_id.

    Two layers are considered duplicates when they have the same ONNX model
    signature (input/output shapes, op types, initializers) within the same
    model scope. The first-seen layer for a given signature becomes the source.
    """
    scope_sig_to_source = {}
    layer_id_to_source = {}
    for case in cases:
        layer_id = _get_layer_id_from_case(case)
        scope = case.name.split("_layer_")[0]
        model = case.data.model if hasattr(case.data, 'model') else case.data
        sig = json.dumps(model_signature(model))
        key = (scope, sig)
        if key not in scope_sig_to_source:
            scope_sig_to_source[key] = layer_id
        else:
            layer_id_to_source[layer_id] = scope_sig_to_source[key]
    return layer_id_to_source


def _copy_result_from_source_layer(layer_id: str, executor: str, source_layer_id: str) -> None:
    """Copy executor result from source layer to duplicate layer.

    If the source layer has a result for this executor, copy it into
    _discovery_state and log. If not, skip the test.
    """
    source_results = _discovery_state.results.get(source_layer_id, {})
    if executor in source_results:
        src = source_results[executor]
        _discovery_state.record_result(
            layer_id,
            executor,
            src["status"],
            src.get("tolerance_used", DEFAULT_TOLERANCE.copy()),
            src.get("max_diff"),
            src.get("failure_report"),
            src.get("timing"),
        )
        _discovery_log(
            f"\nLayer {layer_id}: {executor.upper()} = {src['status']} "
            f"(copied from {source_layer_id})"
        )
        return

    pytest.skip(
        f"Layer {layer_id} is a duplicate of {source_layer_id}, "
        f"but {source_layer_id} was not tested on {executor.upper()}"
    )


def _maybe_skip_executor(
    request,
    layer_id: str,
    executor: str,
    model_name: Optional[str],
    subgraph_suffix: Optional[str] = None,
) -> None:
    """Skip remaining executors for a layer if one already succeeded.

    This is the "skip mode" functionality that speeds up discovery by not
    testing additional executors once one works for a layer.

    Args:
        request: Pytest request object
        layer_id: ID of the layer being tested
        executor: Current executor being tested (nss/css/host)
        model_name: Name of the model
        subgraph_suffix: Optional suffix for subgraph-specific JSON
    """
    # Check if this executor should be skipped entirely (--skip-executors option)
    skipped_executors = _get_skipped_executors(request.config)
    if executor in skipped_executors:
        pytest.skip(f"Executor '{executor}' is in --skip-executors list")

    skip_mode = request.config.getoption("--executor-skip-mode", default=False)
    if not skip_mode or not layer_id or not model_name:
        return

    # First check in-memory results from current test session (e.g., NSS just passed)
    if layer_id in _discovery_state.results:
        for exec_name, exec_result in _discovery_state.results[layer_id].items():
            if exec_result.get("status") == "success":
                pytest.skip(f"Layer {layer_id} already works with {exec_name}, skipping {executor}")

    # Then check persisted results from JSON file (subgraph-specific or main model)
    json_path = _get_json_path(request.config, model_name, subgraph_suffix)
    if not json_path or not json_path.exists():
        return

    existing_data = _load_json(request.config, model_name, subgraph_suffix)
    if layer_id not in existing_data.get("ops", {}):
        return

    op_data = existing_data["ops"][layer_id]
    for exec_name, exec_result in op_data.get("executors", {}).items():
        if exec_result.get("status") == "success":
            pytest.skip(f"Layer {layer_id} already works with {exec_name}, skipping {executor}")


def _get_json_path(
    config, model_name: Optional[str] = None, subgraph_suffix: Optional[str] = None
) -> Path:
    """Get path to executor assignments JSON file.

    Args:
        config: pytest config
        model_name: Base model name
        subgraph_suffix: Optional suffix for subgraph-specific JSON (e.g., "subgraph_0_5")
    """
    output_dir = config.getoption("--executor-discovery-output", default=None)
    if model_name is None:
        model_path = config.getoption("--model-path", default=None)
        model_name = Path(model_path).stem if model_path else "auto_discovered"

    if subgraph_suffix:
        filename = f"executor_assignments_{model_name}_{subgraph_suffix}.json"
    else:
        filename = f"executor_assignments_{model_name}.json"

    return Path(output_dir) / filename if output_dir else Path(filename)


def _load_json(config, model_name: str, subgraph_suffix: Optional[str] = None) -> Dict:
    """Load JSON config for a model."""
    json_path = _get_json_path(config, model_name, subgraph_suffix)
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            _discovery_log(
                f"[ExecutorJSON] Loaded {len(data.get('ops', {}))} op(s) from {json_path}"
            )
            return data
        except Exception as e:
            _discovery_log(f"[ExecutorJSON] Error loading {json_path}: {e}")
    return {
        "version": "1.1",
        "default_tolerance": DEFAULT_TOLERANCE.copy(),
        "ops": {},
        "model_name": model_name,
    }


def _save_json(config, model_name: str, data: Dict) -> None:
    """Save JSON config for a model."""
    json_path = _get_json_path(config, model_name)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def _get_tolerance(layer_id: str, json_data: Dict) -> Dict[str, float]:
    """Get tolerance for an operation from JSON data."""
    ops = json_data.get("ops", {})
    if layer_id in ops:
        op_data = ops[layer_id]
        # Check executor-specific tolerance first
        for executor in EXECUTOR_ORDER:
            exec_data = op_data.get("executors", {}).get(executor, {})
            if "tolerance_used" in exec_data:
                return exec_data["tolerance_used"]
        # Fall back to op's default tolerance
        if "tolerance_used" in op_data:
            return op_data["tolerance_used"]
    return json_data.get("default_tolerance", DEFAULT_TOLERANCE.copy())


def _extract_model_name_from_case(case) -> Optional[str]:
    """Extract model name from case name."""
    # Check subgraph first (subgraph cases also contain _layer_)
    if "_subgraph_" in case.name:
        return case.name.split("_subgraph_")[0]
    elif "_layer_" in case.name:
        return case.name.split("_layer_")[0]
    elif "_full_model" in case.name:
        return case.name.split("_full_model")[0]
    return None


def _is_subgraph_case(case) -> bool:
    """Check if case is a subgraph case."""
    return "_subgraph_" in case.name


def _get_subgraph_suffix(case) -> Optional[str]:
    """Extract subgraph suffix from case name (e.g., 'subgraph_0_5')."""
    if "_subgraph_" not in case.name:
        return None
    # Case name format: {model}_subgraph_{from}_{to}_layer_... or {model}_subgraph_{from}_{to}_full
    # Extract only the subgraph_{from}_{to} part
    parts = case.name.split("_subgraph_")
    if len(parts) >= 2:
        # Get the part after _subgraph_, which starts with {from}_{to}
        after_subgraph = parts[1]
        # Split by _ and take only the first two numeric parts (from_index, to_index)
        subparts = after_subgraph.split("_")
        if len(subparts) >= 2:
            return f"subgraph_{subparts[0]}_{subparts[1]}"
    return None


def _get_recommended_executor(executors: Dict[str, Any], recommend_by_timing: bool = False) -> Optional[str]:
    """Get the recommended executor from discovery results.

    Priority (when recommend_by_timing is False):
    1. First working executor (status == "success")
    2. First difference executor (status == "difference") if no success
    3. None - if all error or no results

    Priority (when recommend_by_timing is True):
    1. Fastest executor with status == "success" (based on timing.runtime_ms)
    2. Fastest executor with status == "difference" if no success
    3. None - if all error or no results

    Executor order: nss -> css -> host (when not using timing)
    """
    if recommend_by_timing:
        # Find fastest success executor
        fastest_success = None
        fastest_success_time = float("inf")
        for executor in EXECUTOR_ORDER:
            if executor in executors and executors[executor].get("status") == "success":
                timing = executors[executor].get("timing", {})
                runtime_ms = timing.get("runtime_ms") if timing else None
                if runtime_ms is not None and runtime_ms < fastest_success_time:
                    fastest_success_time = runtime_ms
                    fastest_success = executor
        if fastest_success:
            return fastest_success

        # Find fastest difference executor if no success
        fastest_diff = None
        fastest_diff_time = float("inf")
        for executor in EXECUTOR_ORDER:
            if executor in executors and executors[executor].get("status") == "difference":
                timing = executors[executor].get("timing", {})
                runtime_ms = timing.get("runtime_ms") if timing else None
                if runtime_ms is not None and runtime_ms < fastest_diff_time:
                    fastest_diff_time = runtime_ms
                    fastest_diff = executor
        if fastest_diff:
            return fastest_diff
    else:
        # First priority: find first success
        for executor in EXECUTOR_ORDER:
            if executor in executors and executors[executor].get("status") == "success":
                return executor

        # Second priority: find first difference
        for executor in EXECUTOR_ORDER:
            if executor in executors and executors[executor].get("status") == "difference":
                return executor

    # All error or no results - return None
    return None


def _update_json_with_results(json_data: Dict, mlir_file: Optional[Path], recommend_by_timing: bool = False) -> None:
    """Update JSON data with discovery results and line numbers."""
    # Ensure ops dict exists
    if "ops" not in json_data:
        json_data["ops"] = {}

    # Update with discovery results
    for layer_id, executors in _discovery_state.results.items():
        if layer_id not in json_data["ops"]:
            json_data["ops"][layer_id] = {"executors": {}}

        # Update executors
        for executor, result in executors.items():
            json_data["ops"][layer_id]["executors"][executor] = result

        # Always save node_index for later use by full model test
        node_index = _discovery_state.node_indices.get(layer_id)
        if node_index is not None:
            json_data["ops"][layer_id]["_node_index"] = node_index

        # Use the pre-computed full MLIR location if available
        full_mlir_location = _discovery_state.full_mlir_locations.get(layer_id)
        if full_mlir_location and re.match(r"^\d+:\d+$", full_mlir_location):
            json_data["ops"][layer_id]["mlir_location"] = full_mlir_location
        elif layer_id in _discovery_state.locations:
            # Fallback to recorded location (op_type)
            json_data["ops"][layer_id]["mlir_location"] = _discovery_state.locations[layer_id]

    # Recalculate recommended_executor for all ops (including existing ones)
    for layer_id, op_data in json_data["ops"].items():
        executors = op_data.get("executors", {})
        recommended = _get_recommended_executor(executors, recommend_by_timing)
        if recommended:
            json_data["ops"][layer_id]["recommended_executor"] = recommended
        elif executors:
            # Has executors but none successful - set to None explicitly
            json_data["ops"][layer_id]["recommended_executor"] = None


def _save_discovery_results(
    config, case, mlir_file: Optional[Path], is_subgraph: bool = False
) -> None:
    """Save discovery results to JSON file.

    For subgraph cases, saves to a subgraph-specific JSON file.
    """
    model_name = _extract_model_name_from_case(case) if case else None
    if not model_name:
        return

    # Get recommend_by_timing option
    recommend_by_timing = config.getoption("--recommend-by-timing", default=False)

    # Determine JSON path (subgraph-specific or main model)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    json_data = _load_json(config, model_name, subgraph_suffix)
    _update_json_with_results(json_data, mlir_file, recommend_by_timing)

    if model_name:
        json_data["model_name"] = model_name

    json_name = model_name if not subgraph_suffix else f"{model_name}_{subgraph_suffix}"
    _save_json(config, json_name, json_data)

    # Get summary and critical failures once
    summary = _discovery_state.get_summary()
    critical_failures = _get_all_critical_failures()

    # Print summary
    _discovery_log("\n\nExecutor Discovery Summary")
    output_path = _get_json_path(config, model_name, subgraph_suffix)
    _discovery_log(f"Output file: {output_path}")
    _discovery_log(f"Total layers: {summary['total_layers']}")
    _discovery_log("\nStatus counts:")
    for status, count in summary["status_counts"].items():
        _discovery_log(f"  {status}: {count}")
    _discovery_log("\nFirst working executor distribution:")
    for executor in EXECUTOR_ORDER:
        count = summary["executor_counts"].get(executor, 0)
        _discovery_log(f"  {executor.upper()}: {count}")

    # Print critical failures if any layer has all executors failed
    if critical_failures:
        msg = f"\nCRITICAL: {len(critical_failures)} layer(s) where ALL executors have ERRORS!"
        _discovery_log(msg)
        for cf in critical_failures:
            _discovery_log(f"  - {cf['layer_id']} (node: {cf['node_index']})")
        _discovery_log(f"\nThese layers CANNOT work. Check the JSON for details.")
    _discovery_log("")

    # Save detailed report with critical failures info
    _save_detailed_report(config, model_name, critical_failures, summary)


def _extract_max_diff(error_msg: str) -> Optional[Dict[str, float]]:
    """Extract max differences from comparison error message."""
    metrics = parse_diff_metrics(error_msg)
    if not metrics:
        return None
    # Return only the float-valued diff keys
    return {
        k: v for k, v in metrics.items()
        if k in ("max_rel_diff", "max_abs_diff") and isinstance(v, float)
    } or None


def _extract_failure_report(error_msg: str, error_type: type) -> Dict[str, str]:
    """Extract failure report from error message."""
    if error_type == AssertionError:
        metrics = parse_diff_metrics(error_msg)
        parts = []
        if "max_rel_diff" in metrics:
            parts.append(f"Max relative difference: {metrics['max_rel_diff']}")
        if "max_abs_diff" in metrics:
            parts.append(f"Max absolute difference: {metrics['max_abs_diff']}")
        if "num_differences" in metrics:
            parts.append(
                f"Number of differences: {metrics['num_differences']} out of "
                f"{metrics['total_elements']} [{metrics['diff_percentage']}%]"
            )
        return {
            "type": "accuracy_failure",
            "summary": "\n".join(parts) if parts else "Accuracy check failed",
        }
    # Other errors
    return {"type": "error", "summary": f"{error_type.__name__}: Test execution failed"}


def _resolve_op_name_to_index(op_name, name_to_index, option_name):
    """Resolve an op name to its node index."""
    if op_name in name_to_index:
        return name_to_index[op_name]
    available = list(name_to_index.keys())[:5]
    raise ValueError(
        f"Could not resolve {option_name}='{op_name}'. Available: {available}..."
    )


def _discover_model_files(config) -> List[Path]:
    """Discover ONNX model files from --model-path or default directories."""
    onnx_model_path = config.getoption("--model-path", default=None)
    if onnx_model_path:
        path = Path(onnx_model_path)
        if not path.exists():
            raise pytest.UsageError(f"--model-path does not exist: {onnx_model_path}")
        return [path]
    files = list_files("dev_ops", ".onnx", False) + list_files("onnx_models", ".onnx", False)
    if not files:
        pytest.skip("No ONNX models found")
    return files


def _precompute_mlir_mappings(files: List[Path], config) -> Dict[str, Dict[str, str]]:
    """Pre-compute ONNX-to-full-MLIR line-number mappings for each model."""
    onnx_to_mlir_map = {}
    if not hasattr(pytest_generate_tests, '_onnx_to_mlir_maps'):
        pytest_generate_tests._onnx_to_mlir_maps = {}

    for f in files:
        model_name = f.stem
        json_path = _get_json_path(config, model_name)
        mlir_cache = Path(f".pytest_cache/onnx_full_mlir/{model_name}.mlir")

        if model_name not in pytest_generate_tests._onnx_to_mlir_maps:
            mapping = _build_onnx_to_mlir_mapping(f, mlir_cache)
            pytest_generate_tests._onnx_to_mlir_maps[model_name] = mapping

            # Pre-populate JSON with mapping if it doesn't exist
            if json_path and not json_path.exists():
                json_data = {
                    "version": "1.1",
                    "model_name": model_name,
                    "default_tolerance": DEFAULT_TOLERANCE.copy(),
                    "ops": {},
                    "_onnx_to_mlir_map": mapping,
                }
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, "w") as jf:
                    json.dump(json_data, jf, indent=2)

        onnx_to_mlir_map[model_name] = pytest_generate_tests._onnx_to_mlir_maps[model_name]

    return onnx_to_mlir_map


def _maybe_apply_bf16_conversion(model, f: Path, auto_convert: bool, save_path: Optional[str]) -> Any:
    """Apply BF16 conversion to a model if requested.

    Returns the (possibly converted) model.
    """
    if not auto_convert:
        return model
    from torq.testing.onnx import convert_fp32_to_bf16, is_model_bf16
    if is_model_bf16(model):
        _discovery_log(f"[BF16] Model {f.name} already in BF16 format")
        return model
    _discovery_log(f"[BF16] Converting {f.name} to BF16...")
    converted = convert_fp32_to_bf16(model)
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(converted, str(sp))
        _discovery_log(f"[BF16] Saved converted model to: {sp}")
    return converted


def _generate_subgraph_cases(f: Path, config) -> List[Case]:
    """Generate test cases in subgraph mode."""
    from torq.testing.onnx import get_full_model

    subgraph_from = config.getoption("--subgraph-from")
    subgraph_to = config.getoption("--subgraph-to")
    auto_convert_bf16 = config.getoption("--auto-convert-bf16", default=False)
    save_bf16_path = config.getoption("--save-bf16-model", default=None)
    model_name = f.stem

    full_model = get_full_model(str(f))
    full_model = _maybe_apply_bf16_conversion(full_model, f, auto_convert_bf16, save_bf16_path)

    # Build name -> index mapping and resolve op names to indices
    all_layers = generate_onnx_layers_from_model(full_model, node_groups=None, dedup=False)
    name_to_index = {}
    for layer_data in all_layers.values():
        layer_node_index = getattr(layer_data, 'node_index', None)
        if layer_node_index is None:
            continue
        model = layer_data.model if hasattr(layer_data, 'model') else layer_data
        graph = model.graph
        if not graph.node:
            continue
        op_type = graph.node[0].op_type
        output_name = graph.node[0].output[0]
        name_to_index[f"{op_type}_{output_name}"] = layer_node_index
        name_to_index[output_name] = layer_node_index

    from_index = _resolve_op_name_to_index(subgraph_from, name_to_index, "--subgraph-from")
    to_index = _resolve_op_name_to_index(subgraph_to, name_to_index, "--subgraph-to")
    _discovery_log(f"[Subgraph] Resolved '{subgraph_from}' -> {from_index}, "
                   f"'{subgraph_to}' -> {to_index}")

    _discovery_log(f"[Subgraph] Extracting subgraph from node {from_index} to "
                   f"{to_index} from {f.name}")
    try:
        subgraph = extract_onnx_subgraph(full_model, from_index, to_index)
    except Exception as e:
        _discovery_log(f"[Subgraph] Error extracting subgraph: {e}")
        pytest.skip(f"Failed to extract subgraph: {e}")
        return []

    subgraph_suffix = f"subgraph_{from_index}_{to_index}"
    node_count = len(subgraph.model.graph.node)
    _discovery_log(f"[Subgraph] Successfully extracted subgraph with {node_count} nodes")

    # Extract layers FROM the subgraph (like a mini full-model workflow)
    subgraph_layers = generate_onnx_layers_from_model(
        subgraph.model, node_groups=None, dedup=False
    )
    _discovery_log(f"[Subgraph] Extracted {len(subgraph_layers)} layers from subgraph")

    cases = []
    for layer_key, layer_data in subgraph_layers.items():
        case_name = f"{model_name}_{subgraph_suffix}_{layer_key}"
        cases.append(Case(case_name, layer_data))
    cases.append(Case(f"{model_name}_{subgraph_suffix}_full", subgraph))
    return cases


def _generate_layer_cases(f: Path, config) -> List[Case]:
    """Generate test cases in normal layer-extraction mode."""
    auto_convert_bf16 = config.getoption("--auto-convert-bf16", default=False)
    save_bf16_path = config.getoption("--save-bf16-model", default=None)

    if auto_convert_bf16:
        from torq.testing.onnx import get_full_model
        model = get_full_model(str(f))
        model = _maybe_apply_bf16_conversion(model, f, auto_convert_bf16, save_bf16_path)
        layers = generate_onnx_layers_from_model(model, node_groups=None, dedup=False)
        return [
            Case(f"{f.stem}_{key}", layer)
            for key, layer in layers.items()
        ] + [Case(f"{f.stem}_full_model", model)]
    return generate_onnx_layers_from_file(f)


def _assemble_layer_test_cases(
    cases: List[Case],
    onnx_to_mlir_map: Dict[str, Dict[str, str]],
    skipped_executors: set,
    dedup_layers: bool,
) -> List[tuple]:
    """Build param tuples for per-layer / per-executor parametrization."""
    non_full_cases = [
        c for c in cases
        if "_full_model" not in c.name and not c.name.endswith("_full")
    ]

    layer_id_to_source = _build_duplicate_layer_map(non_full_cases) if dedup_layers else {}

    test_cases = []
    for case in non_full_cases:
        model_wrapper = case.data
        node_index = getattr(model_wrapper, 'node_index', None)

        if "_subgraph_" in case.name:
            model_name = case.name.split("_subgraph_")[0]
            is_subgraph = True
        else:
            model_name = case.name.split("_layer_")[0]
            is_subgraph = False

        layer_id = _get_layer_id_from_case(case)
        full_mlir_location = (
            onnx_to_mlir_map.get(model_name, {}).get(layer_id) if model_name else None
        )
        source_layer_id = layer_id_to_source.get(layer_id)

        for executor in EXECUTOR_ORDER:
            if executor in skipped_executors:
                continue
            test_cases.append(
                (case, layer_id, executor, node_index, full_mlir_location, is_subgraph, source_layer_id)
            )

    # Full model / full subgraph tests
    for case in [c for c in cases if "_full_model" in c.name or c.name.endswith("_full")]:
        test_cases.append((case, None, "discovered", None, None, False, None))

    return test_cases


def pytest_generate_tests(metafunc):
    """Generate test cases for each layer, subgraph, or full model."""
    files = _discover_model_files(metafunc.config)
    if not files:
        return

    skipped_executors = _get_skipped_executors(metafunc.config)
    if skipped_executors:
        _discovery_log(f"[SkipExecutors] Will skip executors: {sorted(skipped_executors)}")

    onnx_to_mlir_map = _precompute_mlir_mappings(files, metafunc.config)

    subgraph_mode = (
        metafunc.config.getoption("--subgraph-from", default=None) is not None
        and metafunc.config.getoption("--subgraph-to", default=None) is not None
    )

    cases = []
    for f in files:
        if subgraph_mode:
            cases += _generate_subgraph_cases(f, metafunc.config)
        else:
            cases += _generate_layer_cases(f, metafunc.config)

    dedup_layers = metafunc.config.getoption("--dedup-layers", default=False)
    test_cases = _assemble_layer_test_cases(cases, onnx_to_mlir_map, skipped_executors, dedup_layers)

    metafunc.parametrize(
        "layer_executor_case",
        test_cases,
        indirect=True,
        ids=[f"{c.name}_{exec}" for c, _, exec, _, _, _, _ in test_cases],
    )


@pytest.fixture
def reference_results(request, onnx_layer_model):
    return request.getfixturevalue("numpy_reference_results")


@pytest.fixture
def layer_executor_case(request):
    """Provide the layer case with executor info."""
    case, layer_id, executor, node_index, full_mlir_location, is_subgraph, source_layer_id = request.param
    return {
        "case": case,
        "layer_id": layer_id,
        "executor": executor,
        "node_index": node_index,
        "full_mlir_location": full_mlir_location,
        "is_subgraph": is_subgraph,
        "source_layer_id": source_layer_id,
    }


@pytest.fixture
def onnx_layer_model(request, layer_executor_case):
    """Provide the layer model."""
    case = layer_executor_case["case"]
    version = "onnx_layer_" + case.name
    return VersionedUncachedData(data=case.data, version=version)


@versioned_hashable_object_fixture
def torq_compiler_options(request, case_config):
    """Override torq_compiler_options to add executor map for full model tests."""
    # Start with the compiler options from case_config
    cmds = case_config.get("torq_compiler_options", [])

    if request.config.getoption("--extra-torq-compiler-options"):
        cmds.extend(request.config.getoption("--extra-torq-compiler-options").split(" "))

    if request.config.getoption("--trace-buffers"):
        cmds.append("--torq-enable-buffer-debug-info")

    gdb_port = request.config.getoption("--debug-torq-compiler")
    if gdb_port > 0:
        cmds = ["gdbserver", "localhost:" + str(gdb_port)] + cmds

    # Only add executor map for full model tests (discovered executor mode)
    # and only if case_config doesn't already provide one
    # (Layer tests provide their own temporary executor maps)
    has_executor_map = any("--torq-executor-map=" in str(opt) for opt in cmds)
    executor = request.getfixturevalue("layer_executor_case")["executor"]

    if not has_executor_map and executor == "discovered":
        try:
            torq_executor_assignments_json = request.getfixturevalue(
                "torq_executor_assignments_json"
            )
            if (
                torq_executor_assignments_json
                and torq_executor_assignments_json.file_path.exists()
            ):
                cmds.append(
                    f"--torq-executor-map={torq_executor_assignments_json.file_path}"
                )
        except pytest.FixtureLookupError:
            pass

    return cmds


@pytest.fixture
def comparison_config_for_executor_discovery(request, layer_executor_case):
    """Provide comparison config with per-op tolerances."""
    layer_id = layer_executor_case["layer_id"]
    case = layer_executor_case["case"]
    is_subgraph = layer_executor_case.get("is_subgraph", False)
    model_name = _extract_model_name_from_case(case)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    json_data = _load_json(request.config, model_name, subgraph_suffix) if model_name else {}
    tolerance = _get_tolerance(layer_id, json_data)

    return {
        "int_tol": 1,
        "int_thld": 1,
        "fp_avg_tol": tolerance.get("fp_avg_tol", 0.01),
        "fp_max_tol": tolerance.get("fp_max_tol", 0.01),
        "epsilon": 1e-6,
        "allow_all_zero": False,
        "skip_nan_check": False,
    }


@pytest.fixture(autouse=True)
def save_progress(request, layer_executor_case):
    """Save discovery progress after each test."""
    yield

    # MLIR file is now recorded during test execution to avoid fixture teardown issues
    case = layer_executor_case.get("case") if layer_executor_case else None
    is_subgraph = layer_executor_case.get("is_subgraph", False)
    if case:
        _save_discovery_results(request.config, case, None, is_subgraph)



# Final Reporting
def _get_all_critical_failures() -> List[Dict[str, Any]]:
    """Identify layers where ALL executors have ERROR status (setup/compilation failures).

    "difference" means executor is working but results don't match - NOT a critical failure.
    "error" means setup/compilation/runtime error - IS a critical failure.

    Returns list of critical failure records.
    """
    critical_failures = []

    for layer_id, executors in _discovery_state.results.items():
        tested_executors = [e for e in EXECUTOR_ORDER if e in executors]
        if not tested_executors:
            continue

        all_error = True
        error_details = {}

        for executor in tested_executors:
            status = executors[executor].get("status", "error")
            if status in ("success", "difference"):
                # At least one tested executor is working
                all_error = False
                break
            # Only "error" status counts as critical
            error_details[executor] = {
                "status": status,
                "report": executors[executor].get("failure_report", {}),
            }

        if all_error:
            critical_failures.append({
                "layer_id": layer_id,
                "error_details": error_details,
                "node_index": _discovery_state.node_indices.get(layer_id),
                "mlir_location": _discovery_state.locations.get(layer_id),
                "full_mlir_location": _discovery_state.full_mlir_locations.get(layer_id),
            })

    return critical_failures


def _generate_report_sections(
    model_name: str, summary: Dict, critical_failures: List[Dict]
) -> Dict[str, List[str]]:
    """Generate report sections as reusable lists of lines."""
    sections = {
        "header": [
            "FINAL EXECUTOR DISCOVERY REPORT",
            "",
            f"Model: {model_name}",
            f"Total layers tested: {summary['total_layers']}",
        ],
        "status_summary": ["Status Summary:"] + [
            f"  {status}: {count}" for status, count in summary["status_counts"].items()
        ],
        "executor_distribution": ["First Working Executor Distribution:"] + [
            f"  {executor.upper()}: {summary['executor_counts'].get(executor, 0)}"
            for executor in EXECUTOR_ORDER
        ],
        "critical_failures": [],
        "per_layer_status": ["Per-Layer Status:"],
        "full_model_comparison": [],
    }

    # Critical failures section
    if critical_failures:
        sections["critical_failures"].append(
            f"CRITICAL: {len(critical_failures)} layer(s) with ALL executors ERRORS"
        )
        sections["critical_failures"].append("")
        for cf in critical_failures:
            sections["critical_failures"].append(f"  - {cf['layer_id']} (node: {cf['node_index']})")
    else:
        sections["critical_failures"].append(
            "No critical failures - all layers have at least one working executor"
        )

    # Full model comparison metrics
    if _discovery_state.full_model_metrics:
        sections["full_model_comparison"].append("--- Full Model Comparison ---")
        m = _discovery_state.full_model_metrics
        if "max_rel_diff" in m:
            sections["full_model_comparison"].append(f"  Max relative difference: {m['max_rel_diff']}")
        if "max_abs_diff" in m:
            sections["full_model_comparison"].append(f"  Max absolute difference: {m['max_abs_diff']}")
        if "num_differences" in m:
            sections["full_model_comparison"].append(f"  Number of differences: {m['num_differences']}")

    # Per-layer status — sorted by ONNX node index (model order)
    sorted_results = sorted(
        _discovery_state.results.items(),
        key=lambda item: _discovery_state.node_indices.get(item[0], float("inf")),
    )

    rows = []
    for layer_id, executors in sorted_results:
        statuses = {}
        for executor in EXECUTOR_ORDER:
            status = executors.get(executor, {}).get("status", "-")
            statuses[executor] = "-" if status == "unknown" else status
        rows.append({
            "layer_id": layer_id,
            "statuses": statuses,
            "recommended": _get_recommended_executor(executors) or "-",
        })

    sections["per_layer_status"] = format_per_layer_status_table(
        rows, executor_order=EXECUTOR_ORDER, use_color=False
    )

    return sections


def _print_final_report(config):
    """Print comprehensive final report including critical failures."""
    summary = _discovery_state.get_summary()
    critical_failures = _get_all_critical_failures()

    model_path = config.getoption("--model-path", default=None)
    model_name = Path(model_path).stem if model_path else "unknown"
    json_path = _get_json_path(config, model_name)

    sections = _generate_report_sections(model_name, summary, critical_failures)

    # Print header with formatting (always to console, also to log file if configured)
    report_lines = [
        "",
        "",
        "=" * 80,
        sections["header"][0],
        "=" * 80,
    ] + sections["header"][1:] + [
        f"Output file: {json_path}",
        "",
        "--- Status Summary ---",
    ] + sections["status_summary"] + [
        "",
        "--- First Working Executor Distribution ---",
    ] + sections["executor_distribution"]

    if critical_failures:
        report_lines.append(f"\nCRITICAL FAILURES: {len(critical_failures)} layers with ALL executors ERRORS")
        report_lines.append("\nThese layers CANNOT work. Need to fix specific issues:")
        for i, failure in enumerate(critical_failures, 1):
            report_lines.append(f"\n  {i}. Layer: {failure['layer_id']}")
            report_lines.append(f"     Node index: {failure['node_index']}")
            report_lines.append(f"     MLIR location: {failure['mlir_location'] or 'N/A'}")
            loc = failure['full_mlir_location'] or 'N/A'
            report_lines.append(f"     Full MLIR location: {loc}")
            report_lines.append("     Executor errors:")
            for executor, details in failure["error_details"].items():
                status = details["status"]
                report_summary = details["report"].get("summary", "No details")
                summary = report_summary[:100]
                report_lines.append(f"       - {executor.upper()}: {status} - {summary}")
    else:
        report_lines.append("\n--- No Critical Failures ---")
        report_lines.append("All layers have at least one working executor.")

    if sections["full_model_comparison"]:
        report_lines.append("")
        report_lines.extend(sections["full_model_comparison"])

    report_lines.append("\n--- Per-Layer Detailed Status ---")
    report_lines.extend(sections["per_layer_status"])
    report_lines.append("")

    # Colorize per-layer status lines when printing to a terminal
    use_color = sys.stderr.isatty()
    if use_color:
        _COLOR_MAP = {
            "success": "\033[32m",      # Green
            "difference": "\033[33m",   # Yellow
            "error": "\033[31m",        # Red
            "reset": "\033[0m",
        }
        colorized_lines = []
        for line in report_lines:
            # Only colorize lines that look like per-layer status rows
            # (they start with two spaces and have status columns)
            if line.startswith("  ") and not line.startswith("  -") and "Layer" not in line:
                # Replace status words with colored versions
                for status, code in _COLOR_MAP.items():
                    if status in line:
                        line = line.replace(status, f"{code}{status}{_COLOR_MAP['reset']}")
            colorized_lines.append(line)
        report_lines = colorized_lines

    # Print to stderr (goes to console when stdout/stderr are restored,
    # or to log file when they are redirected)
    for line in report_lines:
        print(line, file=sys.stderr)

    _save_detailed_report(config, model_name, critical_failures, summary)


def _generate_final_report_text(
    model_name: str, summary: Dict, critical_failures: List[Dict]
) -> str:
    """Generate final report text for JSON storage."""
    sections = _generate_report_sections(model_name, summary, critical_failures)
    return "\n".join(build_report_lines(sections))


def _save_detailed_report(config, model_name: str, critical_failures: List[Dict], summary: Dict):
    """Save detailed report including critical failures to JSON."""
    json_path = _get_json_path(config, model_name)
    if not json_path:
        return

    try:
        json_data = _load_json(config, model_name) if json_path.exists() else {
            "version": "1.1",
            "model_name": model_name,
            "default_tolerance": DEFAULT_TOLERANCE.copy(),
            "ops": {},
        }

        # Add discovery report with summary and critical failures only
        # (detailed results are in "ops" section)
        json_data["discovery_report"] = {
            "summary": summary,
            "critical_failures": critical_failures,
        }

        # Add flag if model has critical issues
        json_data["has_critical_failures"] = len(critical_failures) > 0
        json_data["final_report_text"] = _generate_final_report_text(
            model_name, summary, critical_failures
        )

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        _discovery_log(f"\nDetailed report saved to: {json_path}")

    except Exception as e:
        _discovery_log(f"\nWarning: Failed to save detailed report: {e}")


def _build_timing_data(runtime_times: List[float]) -> Optional[Dict[str, Any]]:
    """Build timing data from collected runtime measurements.

    Note: Compilation timing is not captured here as compilation happens
    in test fixtures before _run_layer_test() is called.
    """
    if not runtime_times:
        return None

    avg_runtime = sum(runtime_times) / len(runtime_times)
    return {
        "runtime_ms": round(avg_runtime, TIMING_PRECISION),
        "runs": len(runtime_times),
    }


def _run_layer_test(
    request, torq_results, reference_results, case_config,
    layer_id, executor, node_index, mlir_file_path, json_data, case_name
):
    """Run a single layer test with result recording to JSON."""
    from torq.testing.comparison import compare_test_results
    import time

    tolerance_used = _get_tolerance(layer_id, json_data)
    collect_timing = request.config.getoption("--collect-timing", default=False)
    timing_runs = request.config.getoption("--timing-runs", default=1)

    def _record(status, max_diff=None, failure_report=None, timing=None):
        _discovery_state.record_result(
            layer_id, executor, status, tolerance_used, max_diff, failure_report, timing
        )
        if node_index is not None:
            _discovery_state.record_metadata(layer_id, node_index=node_index)

    def _format_timing_str(timing: Optional[Dict]) -> str:
        """Format timing for display."""
        if timing and "runtime_ms" in timing:
            return f" ({timing['runtime_ms']}ms)"
        return ""

    runtime_times = []

    try:
        for _ in range(timing_runs if collect_timing else 1):
            run_start = time.perf_counter()
            try:
                compare_test_results(request, torq_results, reference_results, case_config)
            except AssertionError:
                raise
            finally:
                if collect_timing:
                    runtime_times.append((time.perf_counter() - run_start) * 1000)

        timing = _build_timing_data(runtime_times) if collect_timing else None

        _record("success", timing=timing)

        op_type = case_name.split("_")[-2]
        loc = _extract_op_type_from_layer(mlir_file_path, op_type)
        if loc:
            _discovery_state.record_metadata(layer_id, mlir_location=loc)

        _discovery_log(f"\nLayer {layer_id}: {executor.upper()} = success{_format_timing_str(timing)}")

    except AssertionError as e:
        max_diff = _extract_max_diff(str(e))
        failure_report = _extract_failure_report(str(e), AssertionError)
        timing = _build_timing_data(runtime_times) if collect_timing else None

        _record("difference", max_diff, failure_report, timing)
        _discovery_log(f"\nLayer {layer_id}: {executor.upper()} = difference{_format_timing_str(timing)}")
        if failure_report.get("summary"):
            _discovery_log(f"  {failure_report['summary']}")
        pytest.fail(f"Layer {layer_id} failed on {executor.upper()} with difference")

    except Exception as e:
        failure_report = _extract_failure_report(str(e), type(e))
        timing = _build_timing_data(runtime_times) if collect_timing else None

        _record("error", failure_report=failure_report, timing=timing)
        _discovery_log(f"\nLayer {layer_id}: {executor.upper()} = error")
        # Re-raise the original exception so pytest marks it as ERROR
        # (pytest.fail() would produce FAILED, which is for assertion failures)
        raise


def executor_discovery(
    request,
    torq_results,
    reference_results,
    case_config,
    layer_executor_case,
    onnx_mlir_model_file,
):
    """Core executor discovery implementation."""
    layer_id = layer_executor_case["layer_id"]
    executor = layer_executor_case["executor"]
    node_index = layer_executor_case.get("node_index")
    full_mlir_location = layer_executor_case.get("full_mlir_location")
    is_subgraph = layer_executor_case.get("is_subgraph", False)
    case = layer_executor_case["case"]

    # Determine which JSON to load (subgraph-specific or main model)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    model_name = _extract_model_name_from_case(case)

    # Load existing JSON results on first test (for consistent reporting in skip mode)
    if not hasattr(executor_discovery, "_loaded_existing_json"):
        executor_discovery._loaded_existing_json = {}

    json_key = f"{model_name}_{subgraph_suffix}" if subgraph_suffix else model_name
    if json_key not in executor_discovery._loaded_existing_json:
        executor_discovery._loaded_existing_json[json_key] = True
        if model_name:
            json_data = _load_json(request.config, model_name, subgraph_suffix)
            if json_data:
                _discovery_state.load_from_json(json_data)

    # Full model / full subgraph test: no layer_id, just compare
    if layer_id is None:
        from torq.testing.comparison import compare_test_results
        import io
        import contextlib

        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                compare_test_results(request, torq_results, reference_results, case_config)
        except AssertionError:
            # Parse comparison metrics from captured stdout
            captured = capture.getvalue()
            metrics = {}
            for line in captured.splitlines():
                if line.startswith("Max relative difference:"):
                    metrics["max_rel_diff"] = line.split(":", 1)[1].strip()
                elif line.startswith("Max absolute difference:"):
                    metrics["max_abs_diff"] = line.split(":", 1)[1].strip()
                elif line.startswith("Number of differences:"):
                    metrics["num_differences"] = line.split(":", 1)[1].strip()
            _discovery_state.full_model_metrics = metrics
            raise
        return

    # Layer mode: record results to JSON (either main model or subgraph-specific)
    _discovery_state.record_metadata(
        layer_id=layer_id,
        node_index=node_index,
        full_mlir_location=full_mlir_location,
        mlir_file=(
            Path(onnx_mlir_model_file.file_path)
            if onnx_mlir_model_file and hasattr(onnx_mlir_model_file, 'file_path')
            else None
        ),
    )

    # Short-circuit duplicate layers: copy results from source layer
    source_layer_id = layer_executor_case.get("source_layer_id")
    if source_layer_id:
        _copy_result_from_source_layer(layer_id, executor, source_layer_id)
        return

    json_data = _load_json(request.config, model_name, subgraph_suffix) if model_name else {}

    mlir_file_path = (
        Path(onnx_mlir_model_file.file_path)
        if onnx_mlir_model_file and hasattr(onnx_mlir_model_file, 'file_path')
        else None
    )

    _run_layer_test(
        request, torq_results, reference_results, case_config,
        layer_id, executor, node_index, mlir_file_path, json_data, case.name
    )


