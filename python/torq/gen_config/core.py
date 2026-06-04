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


def generate_compiler_config(discovery_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a minimal compiler config from discovery data.

    The compiler JSON contains only what the C++ ExecutorAssignmentPass needs:
    {"op_assignments": {"line:column": {"executor": "nss"}, ...}}
    """
    op_assignments: Dict[str, Dict[str, str]] = {}
    for layer_id, op_data in discovery_data.get("ops", {}).items():
        mlir_location = op_data.get("mlir_location")
        recommended = op_data.get("recommended_executor")
        if mlir_location and is_line_column_format(mlir_location) and recommended:
            op_assignments[mlir_location] = {"executor": recommended}
    return {"op_assignments": op_assignments}


def save_compiler_config(path: Path, discovery_data: Dict[str, Any]) -> None:
    """Generate and save the compiler-format config from *discovery_data*."""
    compiler_data = generate_compiler_config(discovery_data)
    save_config(path, compiler_data)
    _discovery_log(f"[CompilerJSON] Saved {len(compiler_data['op_assignments'])} assignment(s) to {path}")


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

    for layer_id, op_data in json_data["ops"].items():
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
