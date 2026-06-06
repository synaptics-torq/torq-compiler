# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core reusable logic for TORQ executor config generation.

This module contains pytest-independent helpers for JSON I/O, executor
recommendation, and report formatting so they can be imported by both the
CLI and the pytest-based discovery engine.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_TOLERANCE = {"fp_avg_tol": 0.01, "fp_max_tol": 0.01}
EXECUTOR_ORDER = ["nss", "css", "host"]
TIMING_PRECISION = 3  # Decimal places for timing values


def _discovery_log(msg: str) -> None:
    """Write discovery diagnostic output to stderr."""
    print(msg, file=sys.stderr)


def _opt(config, short: str, legacy: str, default=None):
    """Check short (canonical) option first, then legacy alias."""
    try:
        val = config.getoption(short, default=None)
        if val is not None and val != default:
            return val
    except ValueError:
        pass
    try:
        return config.getoption(legacy, default=default)
    except ValueError:
        return default


def is_line_column_format(location: str) -> bool:
    """Check if location is in 'line:column' format (e.g., '10:10')."""
    return bool(location and re.match(r"^\d+:\d+$", location))


def get_config_path(
    model_name: str,
    output_dir: Optional[str] = None,
    subgraph_suffix: Optional[str] = None,
) -> Path:
    """Get path to executor assignments JSON file."""
    if subgraph_suffix:
        filename = f"torq_gen_config_{model_name}_{subgraph_suffix}.json"
    else:
        filename = f"torq_gen_config_{model_name}.json"
    return Path(output_dir) / filename if output_dir else Path(filename)


def load_config(path: Path) -> Dict[str, Any]:
    """Load executor config JSON from *path*."""
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            _discovery_log(f"[ExecutorJSON] Loaded {len(data.get('ops', {}))} op(s) from {path}")
            return data
        except Exception as e:
            _discovery_log(f"[ExecutorJSON] Error loading {path}: {e}")
    return {
        "version": "1.1",
        "default_tolerance": DEFAULT_TOLERANCE.copy(),
        "ops": {},
    }


def save_config(path: Path, data: Dict[str, Any]) -> None:
    """Save executor config JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_compiler_config_path(
    model_name: str,
    output_dir: Optional[str] = None,
    subgraph_suffix: Optional[str] = None,
) -> Path:
    """Get path to the compiler-format executor assignments JSON file."""
    if subgraph_suffix:
        filename = f"torq_gen_config_{model_name}_{subgraph_suffix}_compiler.json"
    else:
        filename = f"torq_gen_config_{model_name}_compiler.json"
    return Path(output_dir) / filename if output_dir else Path(filename)


def generate_compiler_config(
    discovery_data: Dict[str, Any], model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a minimal compiler config from discovery data.

    The compiler JSON contains only what the C++ ExecutorAssignmentPass needs:
    {"op_assignments": {"line:column": {"executor": "nss"}, ...}}

    If *model_name* is provided it is stored in the JSON so the file can be
    used standalone (e.g. by ``torq-gen-config run`` without a report JSON).
    """
    op_assignments: Dict[str, Dict[str, str]] = {}
    for layer_id, op_data in discovery_data.get("ops", {}).items():
        mlir_location = op_data.get("mlir_location")
        recommended = op_data.get("recommended_executor")
        if mlir_location and is_line_column_format(mlir_location) and recommended:
            op_assignments[mlir_location] = {"executor": recommended}
    result: Dict[str, Any] = {"op_assignments": op_assignments}
    if model_name is not None:
        result["model_name"] = model_name
    return result


def save_compiler_config(
    path: Path, discovery_data: Dict[str, Any], model_name: Optional[str] = None
) -> None:
    """Generate and save the compiler-format config from *discovery_data*."""
    compiler_data = generate_compiler_config(discovery_data, model_name)
    save_config(path, compiler_data)
    _discovery_log(f"[CompilerJSON] Saved {len(compiler_data['op_assignments'])} assignment(s) to {path}")


def _get_json_path(
    config, model_name: Optional[str] = None, subgraph_suffix: Optional[str] = None
) -> Path:
    """Get path to executor assignments JSON file."""
    output_dir = _opt(config, "--output-dir", "--gen-config-output")
    if model_name is None:
        model_path = _opt(config, "--model", "--model-path")
        model_name = Path(model_path).stem if model_path else "auto_discovered"
    return get_config_path(model_name, output_dir, subgraph_suffix)


def _load_json(config, model_name: str, subgraph_suffix: Optional[str] = None) -> Dict:
    """Load JSON config for a model."""
    path = _get_json_path(config, model_name, subgraph_suffix)
    data = load_config(path)
    if not data.get("model_name"):
        data["model_name"] = model_name
    return data


def get_recommended_executor(
    executors: Dict[str, Any], recommend_by_timing: bool = False
) -> Optional[str]:
    """Get the recommended executor from discovery results.

    Priority (when recommend_by_timing is False):
    1. First working executor (status == "success")
    2. First difference executor (status == "difference") if no success
    3. None - if all error or no results

    Priority (when recommend_by_timing is True):
    1. Fastest executor with status == "success" (based on timing.runtime_ms)
    2. Fastest executor with status == "difference" if no success
    3. None - if all error or no results
    """
    if recommend_by_timing:
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
        for executor in EXECUTOR_ORDER:
            if executor in executors and executors[executor].get("status") == "success":
                return executor
        for executor in EXECUTOR_ORDER:
            if executor in executors and executors[executor].get("status") == "difference":
                return executor

    return None


def get_tolerance(layer_id: str, json_data: Dict[str, Any]) -> Dict[str, float]:
    """Get tolerance for an operation from JSON data."""
    ops = json_data.get("ops", {})
    if layer_id in ops:
        op_data = ops[layer_id]
        for executor in EXECUTOR_ORDER:
            exec_data = op_data.get("executors", {}).get(executor, {})
            if "tolerance_used" in exec_data:
                return exec_data["tolerance_used"]
        if "tolerance_used" in op_data:
            return op_data["tolerance_used"]
    return json_data.get("default_tolerance", DEFAULT_TOLERANCE.copy())


def build_timing_data(runtime_times: List[float]) -> Optional[Dict[str, Any]]:
    """Build timing data from collected runtime measurements."""
    if not runtime_times:
        return None
    avg_runtime = sum(runtime_times) / len(runtime_times)
    return {
        "runtime_ms": round(avg_runtime, TIMING_PRECISION),
        "runs": len(runtime_times),
    }


def update_config_with_results(
    json_data: Dict[str, Any],
    results: Dict[str, Dict[str, Dict[str, Any]]],
    node_indices: Dict[str, int],
    locations: Dict[str, str],
    full_mlir_locations: Dict[str, str],
    recommend_by_timing: bool = False,
) -> None:
    """Update JSON data with discovery results and line numbers."""
    if "ops" not in json_data:
        json_data["ops"] = {}

    for layer_id, executors in results.items():
        if layer_id not in json_data["ops"]:
            json_data["ops"][layer_id] = {"executors": {}}

        for executor, result in executors.items():
            json_data["ops"][layer_id]["executors"][executor] = result

        node_index = node_indices.get(layer_id)
        if node_index is not None:
            json_data["ops"][layer_id]["_node_index"] = node_index

        full_mlir_location = full_mlir_locations.get(layer_id)
        if full_mlir_location and re.match(r"^\d+:\d+$", full_mlir_location):
            json_data["ops"][layer_id]["mlir_location"] = full_mlir_location
        elif layer_id in locations:
            json_data["ops"][layer_id]["mlir_location"] = locations[layer_id]

    # Only recompute recommended_executor for ops that were just discovered
    # (present in *results*). Existing ops that the user may have edited
    # manually (e.g. via `torq-gen-config edit`) are left untouched.
    for layer_id in results:
        op_data = json_data["ops"].get(layer_id, {})
        executors = op_data.get("executors", {})
        recommended = get_recommended_executor(executors, recommend_by_timing)
        if recommended:
            json_data["ops"][layer_id]["recommended_executor"] = recommended
        elif executors:
            json_data["ops"][layer_id]["recommended_executor"] = None


def extract_model_name_from_case_name(case_name: str) -> Optional[str]:
    """Extract model name from a case name string."""
    if "_subgraph_" in case_name:
        return case_name.split("_subgraph_")[0]
    elif "_layer_" in case_name:
        return case_name.split("_layer_")[0]
    elif "_full_model" in case_name:
        return case_name.split("_full_model")[0]
    return None


def get_subgraph_suffix_from_case_name(case_name: str) -> Optional[str]:
    """Extract subgraph suffix from case name (e.g., 'subgraph_0_5')."""
    if "_subgraph_" not in case_name:
        return None
    parts = case_name.split("_subgraph_")
    if len(parts) >= 2:
        after_subgraph = parts[1]
        subparts = after_subgraph.split("_")
        if len(subparts) >= 2:
            return f"subgraph_{subparts[0]}_{subparts[1]}"
    return None


def _build_report_from_ops(
    ops: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build summary, critical failures, and per-layer rows from an *ops* dict.

    Returns *(summary, critical_failures, rows)* where *rows* are ready for
    :func:`format_per_layer_status_table`.
    """
    status_counts: Dict[str, int] = {"success": 0, "difference": 0, "error": 0}
    executor_counts: Dict[str, int] = {executor: 0 for executor in EXECUTOR_ORDER}
    critical_failures: List[Dict[str, Any]] = []
    executor_timing: Dict[str, List[float]] = {executor: [] for executor in EXECUTOR_ORDER}

    sorted_ops = sorted(
        ops.items(),
        key=lambda item: item[1].get("_node_index", float("inf")),
    )

    rows: List[Dict[str, Any]] = []
    for layer_id, op_data in sorted_ops:
        executors = op_data.get("executors", {})
        tested_executors = [e for e in EXECUTOR_ORDER if e in executors]

        statuses: Dict[str, str] = {}
        all_error = bool(tested_executors)
        first_success: Optional[str] = None

        for executor in EXECUTOR_ORDER:
            result = executors.get(executor, {})
            status = result.get("status", "unknown")
            statuses[executor] = "-" if status == "unknown" else status
            if status in ("success", "difference"):
                all_error = False
            if status == "success" and first_success is None:
                first_success = executor
            if status in status_counts:
                status_counts[status] += 1

            timing = result.get("timing")
            if timing and "runtime_ms" in timing:
                executor_timing[executor].append(timing["runtime_ms"])

        if first_success:
            executor_counts[first_success] += 1

        recommended = op_data.get("recommended_executor")
        if recommended is None:
            recommended = get_recommended_executor(executors) or "-"

        rows.append(
            {
                "layer_id": layer_id,
                "statuses": statuses,
                "recommended": recommended,
            }
        )

        if all_error and tested_executors:
            error_details: Dict[str, Dict[str, Any]] = {}
            for executor in tested_executors:
                error_details[executor] = {
                    "status": executors[executor].get("status", "error"),
                    "report": executors[executor].get("failure_report", {}),
                }
            critical_failures.append(
                {
                    "layer_id": layer_id,
                    "error_details": error_details,
                    "node_index": op_data.get("_node_index"),
                    "mlir_location": op_data.get("mlir_location"),
                }
            )

    summary: Dict[str, Any] = {
        "total_layers": len(ops),
        "status_counts": status_counts,
        "executor_counts": executor_counts,
    }

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

    return summary, critical_failures, rows


def generate_final_report_text(data: Dict[str, Any]) -> Optional[str]:
    """Regenerate final_report_text from the JSON data dict.

    Computes summary and critical failures from *data['ops']* rather than
    relying on the stored ``discovery_report``, so user edits are reflected.
    """
    from torq.gen_config._utils import build_report_lines, format_per_layer_status_table

    ops = data.get("ops", {})
    if not ops:
        return None

    model_name = data.get("model_name", "unknown")
    summary, critical_failures, rows = _build_report_from_ops(ops)

    sections: Dict[str, List[str]] = {
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

    if critical_failures:
        sections["critical_failures"].append(
            f"CRITICAL: {len(critical_failures)} layer(s) with ALL executors ERRORS"
        )
        sections["critical_failures"].append("")
        for cf in critical_failures:
            sections["critical_failures"].append(
                f"  - {cf['layer_id']} (node: {cf.get('node_index')})"
            )
    else:
        sections["critical_failures"].append(
            "No critical failures - all layers have at least one working executor"
        )

    sections["per_layer_status"] = format_per_layer_status_table(
        rows, executor_order=EXECUTOR_ORDER, use_color=False
    )

    return "\n".join(build_report_lines(sections))
