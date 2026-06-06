# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Report generation for executor discovery results.

Produces the final human-readable report and persists it to JSON.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from torq.gen_config._utils import build_report_lines, format_per_layer_status_table
from torq.gen_config.core import (
    DEFAULT_TOLERANCE,
    EXECUTOR_ORDER,
    _build_report_from_ops,
    _discovery_log,
    _get_json_path,
    _load_json,
    _opt,
)
from torq.gen_config._state import ExecutorDiscoveryState


def _get_all_critical_failures(discovery_state: ExecutorDiscoveryState) -> List[Dict[str, Any]]:
    """Identify layers where ALL executors have ERROR status (setup/compilation failures).

    "difference" means executor is working but results don't match - NOT a critical failure.
    "error" means setup/compilation/runtime error - IS a critical failure.

    Returns list of critical failure records.
    """
    critical_failures = []

    for layer_id, executors in discovery_state.results.items():
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
                "node_index": discovery_state.node_indices.get(layer_id),
                "mlir_location": discovery_state.locations.get(layer_id),
                "full_mlir_location": discovery_state.full_mlir_locations.get(layer_id),
            })

    return critical_failures


def _generate_report_sections(
    model_name: str, discovery_state: ExecutorDiscoveryState
) -> Dict[str, List[str]]:
    """Generate report sections as reusable lists of lines."""
    # Convert live discovery state into the same shape as JSON ops
    ops: Dict[str, Any] = {}
    for layer_id, executors in discovery_state.results.items():
        ops[layer_id] = {
            "executors": executors,
            "_node_index": discovery_state.node_indices.get(layer_id),
            "mlir_location": discovery_state.locations.get(layer_id),
            "recommended_executor": discovery_state.recommended_executors.get(layer_id),
        }

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
    if discovery_state.full_model_metrics:
        sections["full_model_comparison"].append("--- Full Model Comparison ---")
        m = discovery_state.full_model_metrics
        if "max_rel_diff" in m:
            sections["full_model_comparison"].append(f"  Max relative difference: {m['max_rel_diff']}")
        if "max_abs_diff" in m:
            sections["full_model_comparison"].append(f"  Max absolute difference: {m['max_abs_diff']}")
        if "num_differences" in m:
            sections["full_model_comparison"].append(f"  Number of differences: {m['num_differences']}")

    sections["per_layer_status"] = format_per_layer_status_table(
        rows, executor_order=EXECUTOR_ORDER, use_color=False
    )

    return sections


def _print_final_report(config, discovery_state: ExecutorDiscoveryState):
    """Print comprehensive final report including critical failures."""
    summary = discovery_state.get_summary()
    critical_failures = _get_all_critical_failures(discovery_state)

    model_path = _opt(config, "--model", "--model-path")
    model_name = Path(model_path).stem if model_path else "unknown"
    json_path = _get_json_path(config, model_name)

    sections = _generate_report_sections(model_name, discovery_state)

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

    _save_detailed_report(config, model_name, discovery_state)


def _generate_final_report_text(
    model_name: str, discovery_state: ExecutorDiscoveryState
) -> str:
    """Generate final report text for JSON storage."""
    sections = _generate_report_sections(model_name, discovery_state)
    return "\n".join(build_report_lines(sections))


def _save_detailed_report(config, model_name: str, discovery_state: ExecutorDiscoveryState):
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

        # Build summary and critical failures from discovery state
        ops = {}
        for layer_id, executors in discovery_state.results.items():
            ops[layer_id] = {
                "executors": executors,
                "_node_index": discovery_state.node_indices.get(layer_id),
                "mlir_location": discovery_state.locations.get(layer_id),
                "recommended_executor": discovery_state.recommended_executors.get(layer_id),
            }
        summary, critical_failures, _ = _build_report_from_ops(ops)

        # Add discovery report with summary and critical failures only
        # (detailed results are in "ops" section)
        json_data["discovery_report"] = {
            "summary": summary,
            "critical_failures": critical_failures,
        }

        # Add flag if model has critical issues
        json_data["has_critical_failures"] = len(critical_failures) > 0
        json_data["final_report_text"] = _generate_final_report_text(
            model_name, discovery_state
        )

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        _discovery_log(f"\nDetailed report saved to: {json_path}")

    except Exception as e:
        _discovery_log(f"\nWarning: Failed to save detailed report: {e}")
