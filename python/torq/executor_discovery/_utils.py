"""Shared utilities for executor discovery tests."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def extract_line_numbers_from_mlir(
    mlir_file: Path, skip_constants: bool = True
) -> List[Tuple[str, str]]:
    """Extract (op_type, line:column) tuples from MLIR in order of appearance.

    Args:
        mlir_file: Path to MLIR file.
        skip_constants: If True, skip Constant operations.

    Returns:
        List of (op_type, "line:column") tuples.
    """
    if not mlir_file.exists():
        return []

    try:
        content = mlir_file.read_text()
    except Exception:
        return []

    result: List[Tuple[str, str]] = []
    for line_num, line_content in enumerate(content.split("\n"), start=1):
        match = re.search(
            r'torch\.operator\s+"onnx\.([A-Za-z]+)"', line_content, re.IGNORECASE
        )
        if not match:
            continue
        op_type = match.group(1)
        if skip_constants and op_type == "Constant":
            continue
        column = line_content.find("torch.operator") + 1
        if column == 0:
            column = 10
        result.append((op_type, f"{line_num}:{column}"))

    return result


def parse_diff_metrics(error_msg: str) -> Dict[str, Any]:
    """Parse numerical comparison metrics from an error message.

    Returns a dict with keys like 'max_rel_diff', 'max_abs_diff',
    'num_differences', 'total_elements', 'diff_percentage'.
    """
    metrics: Dict[str, Any] = {}

    # Max relative difference — try np.float32(...) first, then standard format
    m = re.search(r"np\.float32\(([\d.eE+-]+)\)", error_msg)
    if not m:
        m = re.search(r"Max relative difference:\s*([\d.eE+-]+)", error_msg)
    if m:
        try:
            metrics["max_rel_diff"] = float(m.group(1))
        except ValueError:
            pass

    # Max absolute difference
    m = re.search(r"Max absolute difference:\s*([\d.eE+-]+)", error_msg)
    if m:
        try:
            metrics["max_abs_diff"] = float(m.group(1))
        except ValueError:
            pass

    # Number of differences: "X out of Y [Z%]"
    m = re.search(r"(\d+)\s+out of\s+(\d+)\s+\[([\d.]+)%\]", error_msg)
    if m:
        metrics["num_differences"] = int(m.group(1))
        metrics["total_elements"] = int(m.group(2))
        metrics["diff_percentage"] = float(m.group(3))

    return metrics


def format_per_layer_status_table(
    rows: List[Dict[str, Any]],
    executor_order: List[str] = None,
    use_color: bool = False,
) -> List[str]:
    """Format a per-layer status table with aligned columns.

    Matches the viewer script output: fixed-width status columns (12 chars)
    with executor names as headers.

    Args:
        rows: List of dicts, each with keys:
            - 'layer_id': str
            - 'statuses': dict of executor -> status str
            - 'recommended': str or None
        executor_order: Order of executors for columns (default: ["nss", "css", "host"]).
        use_color: Whether to inject ANSI color codes for status values.

    Returns:
        List of strings including header, separator, and data rows.
    """
    if executor_order is None:
        executor_order = ["nss", "css", "host"]

    if not rows:
        return ["Per-Layer Status:", "  (no layers)"]

    STATUS_COL_WIDTH = 12

    max_layer_width = max(len(r["layer_id"][:60]) for r in rows)

    lines: List[str] = []
    lines.append("Per-Layer Status:")
    header_parts = [f"{ex.upper():<{STATUS_COL_WIDTH}}" for ex in executor_order]
    lines.append(
        f"  {'Layer':<{max_layer_width}}  {' '.join(header_parts)} {'Recommended'}"
    )
    lines.append("-" * (max_layer_width + STATUS_COL_WIDTH * len(executor_order) + 25))

    _COLOR_MAP = {
        "success": "\033[32m",
        "difference": "\033[33m",
        "error": "\033[31m",
        "reset": "\033[0m",
    }

    for row in rows:
        layer_id = row["layer_id"][:60]
        statuses = row.get("statuses", {})
        recommended = row.get("recommended") or "-"

        status_parts = []
        for executor in executor_order:
            status = statuses.get(executor, "-")
            raw = "-" if status == "unknown" else status
            # Pad first, then inject color codes so ANSI escapes don't break alignment
            padded = f"{raw:<{STATUS_COL_WIDTH}}"
            if use_color and raw in _COLOR_MAP:
                padded = padded.replace(raw, f"{_COLOR_MAP[raw]}{raw}{_COLOR_MAP['reset']}", 1)
            status_parts.append(padded)

        lines.append(
            f"  {layer_id:<{max_layer_width}}  {' '.join(status_parts)} {recommended}"
        )

    return lines


def build_report_lines(sections: Dict[str, List[str]]) -> List[str]:
    """Build flat report lines from section dict.

    Sections are concatenated in order with blank lines between them.
    """
    lines: List[str] = []
    ordered_keys = [
        "header",
        "status_summary",
        "executor_distribution",
        "critical_failures",
        "full_model_comparison",
        "per_layer_status",
    ]
    for key in ordered_keys:
        if key in sections and sections[key]:
            if lines:
                lines.append("")
            lines.extend(sections[key])
    return lines
