"""Generate latency plots from template MLIR profiling CSV data.

This module can be used as a script or imported from tests to generate
plots automatically with configurable X-axis and grouping parameters.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt



# Simple, tweakable colors for each variant.
# Feel free to change these hex values to whatever you like.
VARIANT_COLORS = {
    "nss": "#36bd36",   # green
    "css": "#f54e25",   # orange
    "host": "#1f77b4",  # blue
    
}


def _parse_shape_from_row(row: dict) -> str | None:
    """Return a shape string for a CSV row, or None if not parseable.

    We prefer the explicit "shape" column when it contains a simple
    "dimxdimx..."-style string. If that's not usable (older logs), we fall
    back to parsing the shape out of the "testcase" column
    (e.g. "div-bf16_shape3_1x1x64" -> "1x1x64").
    """

    shape_field = row.get("shape", "")

    def _looks_like_shape(s: str) -> bool:
        parts = s.split("x")
        return bool(s) and all(p.isdigit() for p in parts)

    shape = None
    if _looks_like_shape(shape_field):
        shape = shape_field
    else:
        testcase = row.get("testcase", "")
        idx = testcase.find("_shape")
        if idx != -1:
            suffix = testcase[idx + len("_shape") :]
            # suffix like "3_1x1x64" -> take after first "_"
            if "_" in suffix:
                shape = suffix.split("_", 1)[1]

    if not shape:
        return None

    # We don't enforce rank here; any numeric "1x2x3..." style is accepted.
    return shape


def generate_latency_plots(
    csv_path: Path, 
    output_dir: Path | None = None,
    x_param: str = "shape",
    y_param: str = "variant"
) -> None:
    """Generate PNG plots summarizing latency by configurable parameters.

    Args:
        csv_path: path to profiling summary CSV
        output_dir: directory where PNGs are written (defaults to csv_path.parent)
        x_param: parameter name for X-axis (default: "shape")
        y_param: parameter name for grouping/colors (default: "variant")
        
    Examples:
        # Plot shape (X) vs variant (bars)
        generate_latency_plots(csv, output, x_param="shape", y_param="variant")
        
        # Plot chip (X) vs runner (bars)
        generate_latency_plots(csv, output, x_param="chip", y_param="runner")
    """

    if output_dir is None:
        output_dir = csv_path.parent

    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            # For backward compatibility with shape parsing
            if x_param == "shape":
                x_value = _parse_shape_from_row(r)
                if x_value is None:
                    continue
            else:
                x_value = r.get(x_param, "")
                if not x_value:
                    continue
                    
            try:
                latency_us = float(r["latency_us"])
            except (KeyError, ValueError, TypeError):
                # Skip rows with missing or non-numeric latency values
                continue

            rows.append({
                "testcase": r.get("testcase", ""),
                x_param: x_value,
                y_param: r.get(y_param, ""),
                "latency_us": latency_us,
            })

    if not rows:
        return

    # Grouped vertical bar chart:
    #   X = x_param values
    #   Y = latency_us, with one grouped bar per y_param value

    by_x: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_x[r[x_param]].append(r)

    if not by_x:
        return

    # One color per y_param value
    y_values = sorted({r[y_param] for r in rows})
    color_map = {v: VARIANT_COLORS.get(v, f"#{hash(v) % 0xFFFFFF:06x}") for v in y_values}

    # Sort x values
    if x_param == "shape":
        # Sort shapes by increasing total elements for readability
        def _shape_sort_key(s: str):
            try:
                dims = [int(x) for x in s.split("x")]
                prod = 1
                for d in dims:
                    prod *= d
                return (prod, tuple(dims))
            except ValueError:
                return (float("inf"), (s,))
        x_values_sorted = sorted(by_x.keys(), key=_shape_sort_key)
    else:
        # Lexical sort for other parameters
        x_values_sorted = sorted(by_x.keys())
        
    n_x_values = len(x_values_sorted)

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n_x_values), 4))

    group_width = 0.6
    bar_width = group_width / max(1, len(y_values))

    # Space groups out horizontally
    x_positions = [i * 1.4 for i in range(n_x_values)]

    for i, x_val in enumerate(x_values_sorted):
        x_rows = by_x[x_val]
        base_x = x_positions[i]

        for j, y_val in enumerate(y_values):
            # Find row for this (x_val, y_val), if any
            match = next((r for r in x_rows if r[y_param] == y_val), None)
            if match is None:
                continue

            x_center = base_x - group_width / 2 + (j + 0.5) * bar_width
            ax.bar(
                x_center,
                match["latency_us"],
                width=bar_width,
                color=color_map[y_val],
                label=y_val,
            )

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values_sorted, rotation=45, ha="right")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        uniq_handles.append(h)
        uniq_labels.append(lab)
    if uniq_handles:
        ax.legend(
            uniq_handles,
            uniq_labels,
            title=y_param,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
        )

    ax.set_xlabel(x_param)
    ax.set_ylabel("latency_us")

    # Try to derive a generic testcase name from the first row.
    # Look for a .mlir filename in the testcase string (e.g. "add_bf16.mlir" -> "add_bf16")
    testcase_name = None
    first_tc = rows[0].get("testcase", "")

    if "_" in first_tc:
        testcase_name = first_tc.split("_", 1)[0]

    if testcase_name:
        ax.set_title(f"Latency (us) by shape and variant — {testcase_name}")
    else:
        ax.set_title("Latency (us) by shape and variant")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "latency_by_shape_variant_bar.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated latency bar chart: {out_path}")

    