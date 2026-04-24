import hashlib
import io
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from _pytest._io import TerminalWriter

from torq.executor_discovery._utils import extract_line_numbers_from_mlir
from torq.testing.versioned_fixtures import (
    versioned_generated_file_fixture,
    versioned_hashable_object_fixture,
)

"""
Pytest plugin for ONNX executor discovery.

Registered globally via ``torq.testing``'s ``pytest_plugins``.  Hooks are
no-ops unless executor-discovery options are actually used, so they do not
interfere with other tests.
"""

logger = logging.getLogger("torq.testing.executor_discovery")

# Global state for log file redirection
_discovery_log_file_handle = None
_original_stdout = None
_original_stderr = None
_verbose_mode = False

# Counters for progress percentage when log file is active
_total_tests = 0
_completed_tests = 0


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        default=None,
        help="Path to model file (e.g., encoder.onnx) for discovery tests",
    )
    parser.addoption(
        "--executor-discovery-output",
        default=None,
        help="Directory to save executor discovery JSON output (default: current directory)",
    )
    parser.addoption(
        "--executor-skip-mode",
        action="store_true",
        default=False,
        help="Skip remaining executors for a layer if one succeeds (speeds up discovery)",
    )
    parser.addoption(
        "--skip-executors",
        default=None,
        help="Comma-separated list of executors to skip entirely (e.g., 'nss,css' or 'nss'). "
        "Useful when certain executors are known to fail or crash. "
        "Valid values: nss, css, host. Can be combined with --executor-skip-mode.",
    )
    parser.addoption(
        "--auto-convert-bf16",
        action="store_true",
        default=False,
        help="Automatically convert FP32 ONNX models to BF16 for testing",
    )
    parser.addoption(
        "--save-bf16-model",
        default=None,
        help="Save converted BF16 model to specified path (requires --auto-convert-bf16)",
    )
    parser.addoption(
        "--subgraph-from",
        default=None,
        help="Start op name for subgraph extraction "
        "(e.g., 'mobilenetv20_features_conv0_fwd' or 'Conv_mobilenetv20_features_conv0_fwd')",
    )
    parser.addoption(
        "--subgraph-to",
        default=None,
        help="End op name for subgraph extraction "
        "(e.g., 'mobilenetv20_features_relu0_fwd' or 'Relu_mobilenetv20_features_relu0_fwd')",
    )
    parser.addoption(
        "--collect-timing",
        action="store_true",
        default=False,
        help="Collect compile and runtime timing data for each layer/executor",
    )
    parser.addoption(
        "--timing-runs",
        default=1,
        type=int,
        help="Number of runtime runs for timing average (default: 1)",
    )
    parser.addoption(
        "--recommend-by-timing",
        action="store_true",
        default=False,
        help="Recommend executor with fastest runtime when timing data is available "
        "(requires --collect-timing)",
    )
    parser.addoption(
        "--executor-discovery-log-file",
        default=None,
        help="Path to log file for executor discovery output. "
        "When set, all output (pytest, compiler, runtime) goes to the file and "
        "only the final report is shown on console. Use with -s for best results.",
    )
    parser.addoption(
        "--dedup-layers",
        action="store_true",
        default=False,
        help="Detect duplicate layers by ONNX signature and copy executor results "
        "from the first-seen (canonical) layer instead of re-testing. "
        "Each duplicate still appears in JSON with its own correct line number.",
    )


def pytest_configure(config):
    """Redirect stdout/stderr to log file if --executor-discovery-log-file is specified."""
    global _discovery_log_file_handle, _original_stdout, _original_stderr, _verbose_mode
    log_file_path = config.getoption("--executor-discovery-log-file", default=None)
    _verbose_mode = config.getoption("verbose", default=0) > 0
    if log_file_path:
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _discovery_log_file_handle = open(path, "w")
        _original_stdout = sys.stdout
        _original_stderr = sys.stderr
        sys.stdout = _discovery_log_file_handle
        sys.stderr = _discovery_log_file_handle


def pytest_collection_modifyitems(config, items):
    """Track total test count for progress percentage."""
    global _total_tests
    _total_tests = len(items)


def pytest_runtest_setup(item):
    """Print test name at the start of setup when log file is active and -v is set."""
    if _discovery_log_file_handle and _verbose_mode:
        tw = TerminalWriter(sys.__stderr__)
        tw.write(item.nodeid)
        tw.write(" ")
        tw.write("RUNNING", cyan=True)
        tw.write("\n")


def pytest_runtest_logreport(report):
    """Print test name and outcome to terminal when log file is active.

    Replicates pytest's default terminal output with colors, bypassing the
    log file redirection so users can see test progress on the console.
    Also appends (dup) for duplicate layers.
    """
    is_dup = report.when == "call" and hasattr(report, "source_layer_id")

    if not _discovery_log_file_handle or not _verbose_mode:
        # Without log-file redirection, pytest's terminal reporter handles the
        # main output; we just append (dup) after it.
        if is_dup:
            sys.__stderr__.write(" (dup)")
            sys.__stderr__.flush()
        return

    # Print for call phase (all outcomes), or setup phase for non-passed outcomes
    # (skipped/error tests don't have a call phase)
    should_print = (
        report.when == "call"
        or (report.when == "setup" and report.outcome not in ("passed",))
    )
    if not should_print:
        return

    global _completed_tests
    _completed_tests += 1

    # Replicate pytest's terminal reporter behavior:
    # - setup failures are shown as ERROR, not FAILED
    # - call failures are shown as FAILED
    if report.when == "setup" and report.outcome == "failed":
        display_outcome = "ERROR"
        color_kw = {"red": True}
    else:
        display_outcome = report.outcome.upper()
        color_kw = {
            "passed": {"green": True},
            "failed": {"red": True},
            "error": {"red": True},
            "skipped": {"yellow": True},
        }.get(report.outcome, {})

    pct = int(100 * _completed_tests / _total_tests) if _total_tests else 0

    # Build plain-text line for log file (no color codes)
    plain_line = f"{report.nodeid} {display_outcome}"
    if is_dup:
        plain_line += " (dup)"
    plain_line += f" [{pct:3d}%]\n"

    # Write to log file so it has a complete record
    if _discovery_log_file_handle:
        _discovery_log_file_handle.write(plain_line)
        _discovery_log_file_handle.flush()

    # Write colored version to terminal
    tw = TerminalWriter(sys.__stderr__)
    tw.write(report.nodeid)
    tw.write(" ")
    tw.write(display_outcome, **color_kw)
    if is_dup:
        tw.write(" (dup)")
    tw.write(f" [{pct:3d}%]")
    tw.write("\n")


def pytest_sessionfinish(session, exitstatus):
    """Restore stdout/stderr, print final report, and close log file."""
    global _discovery_log_file_handle, _original_stdout, _original_stderr

    # Capture final report to a buffer (while streams are still redirected).
    # We temporarily swap sys.stderr so _print_final_report writes into the
    # buffer instead of the log file, then we manually write the buffer to
    # the log file.  This avoids printing the report to the terminal twice.
    if _verbose_mode and _discovery_log_file_handle:
        try:
            from torq.executor_discovery.executor_discovery_onnx import (
                _discovery_state,
                _print_final_report,
            )
            if _discovery_state.results:
                buf = io.StringIO()
                old_stderr = sys.stderr
                sys.stderr = buf
                try:
                    _print_final_report(session.config)
                finally:
                    sys.stderr = old_stderr
                _discovery_log_file_handle.write(buf.getvalue())
                _discovery_log_file_handle.flush()
        except Exception:
            pass

    # Restore stdout/stderr so the terminal gets the real output
    if _original_stdout and _original_stderr:
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr

    # Print final report to terminal exactly once
    if _verbose_mode:
        try:
            from torq.executor_discovery.executor_discovery_onnx import (
                _discovery_state,
                _print_final_report,
            )
            if _discovery_state.results:
                _print_final_report(session.config)
        except Exception:
            pass

    # Flush log file (don't close it yet — other plugins like terminalreporter
    # may still hold references and write during their teardown)
    if _discovery_log_file_handle:
        _discovery_log_file_handle.flush()


# Helper Functions

def _get_model_name_from_config(config) -> Optional[str]:
    """Extract model name from config options."""
    model_path = config.getoption("--model-path", default=None)
    return Path(model_path).stem if model_path else None


def _find_discovery_json(
    config, model_name: str, subgraph_suffix: Optional[str] = None
) -> Optional[Path]:
    """Find existing executor discovery JSON file for the model.

    For subgraph tests, looks for subgraph-specific JSON file.
    """
    if not model_name:
        return None

    output_dir = config.getoption("--executor-discovery-output", default=None)
    search_dir = Path(output_dir) if output_dir else Path(".")

    if not search_dir.exists():
        return None

    # For subgraph tests, look for subgraph-specific JSON first
    if subgraph_suffix:
        json_path = search_dir / f"executor_assignments_{model_name}_{subgraph_suffix}.json"
        if json_path.exists():
            return json_path

    # Look for executor_assignments_<model_name>.json
    json_path = search_dir / f"executor_assignments_{model_name}.json"
    if json_path.exists():
        return json_path

    # Search by model_name field inside JSON files
    for json_file in search_dir.glob("executor_assignments_*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            if data.get("model_name") == model_name:
                return json_file
        except Exception:
            continue

    return None


def _get_subgraph_suffix_from_nodeid(nodeid: str) -> Optional[str]:
    """Extract subgraph suffix from pytest node ID.

    Example: '...::test_executor_discovery[shufflenet-v2-12_subgraph_19_22_full...'
    Returns: 'subgraph_19_22'
    """
    if "_subgraph_" not in nodeid:
        return None
    # Extract subgraph_{from}_{to} from node ID
    match = re.search(r'_subgraph_(\d+)_(\d+)_', nodeid)
    if match:
        return f"subgraph_{match.group(1)}_{match.group(2)}"
    return None


def _update_discovery_json_line_numbers(
    discovery_json_path: Path, mlir_file: Path
) -> None:
    """Update discovery JSON with correct line numbers from full model MLIR.

    Layer tests save mlir_location as op_type (e.g., "Tanh"). This updates
    it to line:column format (e.g., "10:10") from full model MLIR.
    Uses node_index (position) for matching.
    """
    if not discovery_json_path.exists() or not mlir_file.exists():
        return

    # Get all operations in order from full model MLIR
    all_ops = extract_line_numbers_from_mlir(mlir_file)
    if not all_ops:
        return

    with open(discovery_json_path, "r") as f:
        data = json.load(f)

    updated = False
    for op_name in sorted(data.get("ops", {}).keys()):
        op_data = data["ops"][op_name]

        # Use node_index if available for precise matching
        node_index = op_data.get("_node_index")
        if node_index is not None and 0 <= node_index < len(all_ops):
            op_type, new_location = all_ops[node_index]
            old_location = op_data.get("mlir_location", "")

            if old_location != new_location:
                op_data["mlir_location"] = new_location
                updated = True
                msg = (
                    f"[ExecutorAssignments] Updated {op_name}: {old_location} -> "
                    f"{new_location} (node_index={node_index})"
                )
                logger.info(msg)
        else:
            # Warn if node_index is out of bounds (indicates a bug in discovery)
            if node_index is not None and node_index >= len(all_ops):
                logger.warning(
                    f"[ExecutorAssignments] node_index {node_index} out of bounds for {op_name} "
                    f"(max: {len(all_ops) - 1}). This indicates a bug in layer discovery. "
                    f"Using fallback matching by op_type."
                )
            # Fallback: try to match by op_type (first occurrence)
            op_type = op_name.split("_")[0] if op_name else None
            for ot, loc in all_ops:
                if ot == op_type:
                    op_data["mlir_location"] = loc
                    updated = True
                    logger.info(f"[ExecutorAssignments] Updated {op_name} (fallback): -> {loc}")
                    break

    if updated:
        with open(discovery_json_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[ExecutorAssignments] Updated discovery JSON: {discovery_json_path}")


# Fixtures

@versioned_hashable_object_fixture
def _discovery_json_version(request):
    """Fixture that provides a versioned hash of the discovery JSON file.

    This ensures the executor assignments are regenerated when the discovery JSON changes.
    """
    model_name = _get_model_name_from_config(request.config)

    # Early return if --model-path is not set
    if not model_name:
        return "no_discovery"

    discovery_json = _find_discovery_json(request.config, model_name)

    if discovery_json and discovery_json.exists():
        with open(discovery_json, "rb") as f:
            content = f.read()
            hash_value = hashlib.sha256(content).hexdigest()[:16]
            return f"discovery_{hash_value}"
    return "no_discovery"


@versioned_generated_file_fixture("json")
def torq_executor_assignments_json(
    request, versioned_file, mlir_model_file, _discovery_json_version
):
    """Generate executor assignments JSON file for compiler.

    Priority:
    1. If executor_assignments_<model>.json exists, use it directly
       (compiler now supports both discovery format and op_assignments format)
    2. For subgraph tests, use executor_assignments_<model>_<subgraph>.json
    3. Otherwise, generate a default empty assignments file

    The compiler accepts both formats:
    - Discovery format:
      {"ops": {"Conv_0": {"recommended_executor": "nss", "mlir_location": "10:10"}}}
    - Compiler format: {"op_assignments": {"10:10": {"executor": "nss"}}}
    """

    # Check if this is a subgraph test
    test_node_id = request.node.nodeid if hasattr(request, "node") else ""
    subgraph_suffix = _get_subgraph_suffix_from_nodeid(test_node_id)

    # Option 1: Find and use discovery JSON directly
    # The compiler now supports discovery format natively, no conversion needed
    model_name = _get_model_name_from_config(request.config)
    discovery_json = _find_discovery_json(request.config, model_name, subgraph_suffix)

    if discovery_json:
        suffix_str = f" ({subgraph_suffix})" if subgraph_suffix else ""
        logger.info(f"[ExecutorAssignments] Using discovery JSON{suffix_str}: {discovery_json}")
        # For subgraph tests, the mlir_model_file is already the subgraph MLIR
        _update_discovery_json_line_numbers(discovery_json, mlir_model_file)
        # Copy discovery JSON to versioned file location for compiler to use
        shutil.copy(discovery_json, versioned_file)
        return versioned_file

    # Option 2: Default empty assignments
    # Only log warning for full model tests when --model-path is set
    is_full_model_mode = "_full_model" in test_node_id or "_full" in test_node_id

    if is_full_model_mode and model_name:
        suffix_str = f" ({subgraph_suffix})" if subgraph_suffix else ""
        logger.warning(f"[ExecutorAssignments] No discovery JSON found for model '{model_name}'{suffix_str}.")
        logger.warning(
            "[ExecutorAssignments] Run discovery: "
            "pytest tests/test_onnx_executor_discovery.py "
            "-v --model-path=<model>.onnx"
        )

    default_assignments = {"op_assignments": {}}
    with open(versioned_file, "w") as f:
        json.dump(default_assignments, f, indent=2)

    return versioned_file


# Hooks for Error Recording

def _get_layer_info_from_item(item) -> Optional[Dict[str, str]]:
    """Extract layer_id, executor, is_subgraph, and source_layer_id from test item.

    The layer_executor_case parameter is a tuple:
    (case, layer_id, executor, node_index, full_mlir_location, is_subgraph, source_layer_id)
    where is_subgraph is a boolean and source_layer_id is the canonical layer for dupes.
    """
    if not hasattr(item, "callspec"):
        return None

    layer_executor_case = item.callspec.params.get("layer_executor_case")
    if not isinstance(layer_executor_case, tuple) or len(layer_executor_case) < 3:
        return None

    result = {
        "layer_id": layer_executor_case[1],
        "executor": layer_executor_case[2],
    }

    # Extract is_subgraph if present (6th element, index 5)
    if len(layer_executor_case) >= 6:
        result["is_subgraph"] = layer_executor_case[5]

    # Extract source_layer_id if present (7th element, index 6)
    if len(layer_executor_case) >= 7:
        result["source_layer_id"] = layer_executor_case[6]

    return result


def _record_setup_error(layer_id, executor, is_subgraph, failure_report):
    """Record setup error for single layer or subgraph layer."""
    try:
        from torq.executor_discovery.executor_discovery_onnx import (
            DEFAULT_TOLERANCE,
            _discovery_state,
        )
    except Exception:
        return

    # Avoid duplicates
    if layer_id in _discovery_state.results and executor in _discovery_state.results[layer_id]:
        return

    _discovery_state.record_result(
        layer_id, executor, "error", DEFAULT_TOLERANCE.copy(), failure_report=failure_report
    )

    prefix = "Subgraph" if is_subgraph else "Layer"
    if _verbose_mode:
        print(f"\n{prefix} {layer_id}: {executor.upper()} = error (crash or timeout)", file=sys.stderr)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Catch setup errors and record them in discovery state.
    Also tag duplicate layers so pytest_runtest_logreport can show (dup)."""
    outcome = yield
    report = outcome.get_result()

    # Tag duplicate layers for logreport display
    layer_info = _get_layer_info_from_item(item)
    if layer_info:
        source_layer_id = layer_info.get("source_layer_id")
        if source_layer_id:
            report.source_layer_id = source_layer_id

    if report.when != "setup" or report.outcome != "failed":
        return

    layer_id = layer_info.get("layer_id") if layer_info else None
    executor = layer_info.get("executor") if layer_info else None
    is_subgraph = layer_info.get("is_subgraph", False) if layer_info else False
    if not (layer_id and executor):
        return

    failure_report = {
        "type": "error",
        "summary": f"Fixture/setup failed: {str(report.longrepr)[:200]}",
    }

    _record_setup_error(layer_id, executor, is_subgraph, failure_report)
