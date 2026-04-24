#!/usr/bin/env python3
"""Pretty viewer for executor discovery JSON files."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Allow importing from torq.executor_discovery when run as standalone script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from torq.executor_discovery._utils import format_per_layer_status_table


def use_color() -> bool:
    """Check if we should use ANSI colors (terminal supports it and not redirected)."""
    return sys.stdout.isatty()


def colorize(status: str) -> str:
    """Add color to status strings (only if output is a terminal)."""
    if not use_color():
        return status
    colors = {
        "success": "\033[32m",  # Green
        "difference": "\033[33m",  # Yellow
        "error": "\033[31m",  # Red
        "-": "",
        "reset": "\033[0m",
    }
    code = colors.get(status, "")
    reset = colors["reset"] if code else ""
    return f"{code}{status}{reset}"


def colorize_padded(status: str, width: int) -> str:
    """Add color to status string and pad to width (accounting for ANSI codes)."""
    colored = colorize(status)
    # ANSI codes don't take up display space, so we pad the raw string
    padding = width - len(status)
    return f"{colored}{padding * ' '}"


def format_timing(timing: Dict[str, Any]) -> str:
    """Format timing data for display."""
    if not timing or "runtime_ms" not in timing:
        return ""
    runtime = timing["runtime_ms"]
    return f"({runtime}ms)"


def print_summary(data: Dict[str, Any]) -> None:
    """Print a nice summary of the discovery results."""
    model_name = data.get("model_name", "unknown")
    report = data.get("discovery_report", {})
    summary = report.get("summary", {})
    critical_failures = report.get("critical_failures", [])
    ops = data.get("ops", {})

    print("=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    # Status counts
    print("\nSTATUS COUNTS:")
    status_counts = summary.get("status_counts", {})
    for status, count in status_counts.items():
        print(f"  {colorize(status)}: {count}")

    # Timing summary if available
    timing_summary = summary.get("timing_summary")
    if timing_summary:
        print("\nTIMING SUMMARY (average runtime):")
        for executor, timing in timing_summary.items():
            avg = timing.get("avg_ms", 0)
            min_ms = timing.get("min_ms", 0)
            max_ms = timing.get("max_ms", 0)
            samples = timing.get("samples", 0)
            print(f"  {executor.upper()}: {avg}ms (min: {min_ms}ms, max: {max_ms}ms, samples: {samples})")

    # Critical failures
    print(f"\nCRITICAL FAILURES (all executors error): {len(critical_failures)}")
    for cf in critical_failures:
        layer_id = cf["layer_id"]
        node_idx = cf.get("node_index", "N/A")
        print(f"  - {layer_id} (node: {node_idx})")

    # Per-layer table (read from ops section)
    sorted_ops = sorted(ops.items(), key=lambda x: x[1].get("_node_index", float('inf')))
    rows = []
    for layer_id, op_data in sorted_ops:
        executors = op_data.get("executors", {})
        statuses = {
            "nss": executors.get("nss", {}).get("status", "-"),
            "css": executors.get("css", {}).get("status", "-"),
            "host": executors.get("host", {}).get("status", "-"),
        }
        rows.append({
            "layer_id": layer_id,
            "statuses": statuses,
            "recommended": op_data.get("recommended_executor", "-"),
        })

    table_lines = format_per_layer_status_table(rows, use_color=use_color())
    print()
    for line in table_lines:
        print(line)

    # Tolerance info
    print(f"\nDEFAULT TOLERANCE:")
    tolerance = data.get("default_tolerance", {})
    for key, val in tolerance.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 80)


def print_layer_details(data: Dict[str, Any], layer_id: str) -> None:
    """Print detailed info for a specific layer."""
    ops = data.get("ops", {})
    if layer_id not in ops:
        print(f"Layer '{layer_id}' not found!")
        available = list(ops.keys())[:10]
        print(f"Available layers: {', '.join(available)}...")
        return

    layer_data = ops[layer_id]
    recommended = layer_data.get('recommended_executor', '-')
    print(f"\nLAYER: {layer_id}")
    print("=" * 80)

    print(f"\nNode Index: {layer_data.get('_node_index', 'N/A')}")
    print(f"MLIR Location: {layer_data.get('mlir_location', 'N/A')}")
    print(f"Recommended Executor: {recommended}")

    print("\nExecutor Results:")
    for executor in ["nss", "css", "host"]:
        exec_data = layer_data.get("executors", {}).get(executor, {})
        status = exec_data.get("status", "unknown")
        timing_str = format_timing(exec_data.get("timing"))
        display = f"{colorize(status)} {timing_str}" if timing_str else colorize(status)
        print(f"\n  [{executor.upper()}]: {display}")

        if "max_diff" in exec_data:
            print(f"    Max Diff: {exec_data['max_diff']}")

        if "failure_report" in exec_data:
            report = exec_data["failure_report"]
            print(f"    Type: {report.get('type', 'N/A')}")
            summary = report.get("summary", "N/A")
            # Wrap long summaries
            if len(summary) > 70:
                print(f"    Summary: {summary[:70]}...")
            else:
                print(f"    Summary: {summary}")

        if "tolerance_used" in exec_data:
            print(f"    Tolerance: {exec_data['tolerance_used']}")

        if "timing" in exec_data:
            timing = exec_data["timing"]
            print(f"    Runtime: {timing['runtime_ms']}ms")
            if timing.get("runs", 1) > 1:
                print(f"    Runs: {timing['runs']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python/torq/executor_discovery/view_discovery_json.py <json_file> [layer_id]")
        print("  json_file: Path to executor_assignments_*.json")
        print("  layer_id: (Optional) Show details for specific layer")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    if len(sys.argv) > 2:
        # Show specific layer details
        print_layer_details(data, sys.argv[2])
    else:
        # Show summary
        print_summary(data)


if __name__ == "__main__":
    main()
