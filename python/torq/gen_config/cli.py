# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI entry point for torq-gen-config.

Provides subcommands to discover, view, edit, and run TORQ executor configurations.
"""

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from torq.gen_config.core import (
    generate_compiler_config,
    generate_final_report_text,
    get_compiler_config_path,
    get_config_path,
    load_config,
    save_config,
)
from torq.gen_config._utils import format_per_layer_status_table
from torq.gen_config.view import print_layer_details, print_summary


DEFAULT_TEST_FILE = "tests/test_onnx_gen_config.py"

# CLI → pytest flag mapping.
# Each entry: (pytest_flag_template, argparse_attr, is_bool).
# Templates use {v} for the value placeholder; bool flags have no placeholder.
_DISCOVER_FLAGS = [
    ("--skip-mode",      "skip_mode",           True),
    ("--skip-executors={v}",      "skip_executors",      False),
    ("--auto-convert-bf16",       "auto_convert_bf16",   True),
    ("--save-bf16-model={v}",     "save_bf16_model",     False),
    ("--subgraph-from={v}",       "subgraph_from",       False),
    ("--subgraph-to={v}",         "subgraph_to",         False),
    ("--collect-timing",          "collect_timing",      True),
    ("--timing-runs={v}",         "timing_runs",         False),
    ("--recommend-by-timing",     "recommend_by_timing", True),
    ("--dedup-layers",            "dedup_layers",        True),
    ("--gen-config-log-file={v}", "log_file",            False),
]

_RUN_FLAGS = [
    ("--auto-convert-bf16",       "auto_convert_bf16",   True),
    ("--debug-ir={v}",            "debug_ir",            False),
    ("--recompute-cache",         "recompute_cache",     True),
    ("--gen-config-log-file={v}", "log_file",            False),
    ("--subgraph-from={v}",       "subgraph_from",       False),
    ("--subgraph-to={v}",         "subgraph_to",         False),
]


def _build_extra_args(args: argparse.Namespace, flag_defs: list) -> List[str]:
    """Build pytest extra_args from CLI args using a data-driven flag map."""
    extra_args: List[str] = []
    for template, attr, is_bool in flag_defs:
        val = getattr(args, attr, None)
        if val is None or val is False:
            continue
        extra_args.append(template if is_bool else template.format(v=val))
    if getattr(args, "extra_options", None):
        extra_args.extend(args.extra_options)
    return extra_args


def _find_project_root() -> Path:
    """Find the torq-compiler-dev project root by looking for tests/."""
    cwd = Path.cwd()
    for path in [cwd] + list(cwd.parents):
        if (path / "tests" / "test_onnx_gen_config.py").exists():
            return path
    return cwd


def _resolve_test_and_model(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve test file path and model path from CLI args."""
    project_root = _find_project_root()
    test_file = args.test_file or str(project_root / DEFAULT_TEST_FILE)
    model_path = args.model
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return test_file, model_path


def _run_pytest(
    test_file: str,
    model_path: str,
    test_filter: str,
    output_dir: Optional[str],
    extra_args: List[str],
) -> int:
    """Run pytest with the given filter and return its exit code."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-v",
        "-k",
        test_filter,
        f"--model-path={model_path}",
    ]
    if output_dir:
        cmd.append(f"--output-dir={output_dir}")
    cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def cmd_discover(args: argparse.Namespace) -> int:
    """Run executor discovery on an ONNX model."""
    try:
        test_file, model_path = _resolve_test_and_model(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    extra_args = ["--recompute-cache"] + _build_extra_args(args, _DISCOVER_FLAGS)

    return _run_pytest(
        test_file=test_file,
        model_path=model_path,
        test_filter="_layer_",
        output_dir=args.output_dir,
        extra_args=extra_args,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full model test with discovered executor assignments."""
    try:
        test_file, model_path = _resolve_test_and_model(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Verify config exists (report JSON or compiler JSON)
    model_name = Path(model_path).stem
    subgraph_mode = bool(args.subgraph_from and args.subgraph_to)

    if subgraph_mode:
        # For subgraph runs, accept any torq_gen_config JSON in the output dir
        search_dir = Path(args.output_dir) if args.output_dir else Path(".")
        has_config = any(search_dir.glob("torq_gen_config_*.json"))
    else:
        config_path = get_config_path(model_name, args.output_dir)
        compiler_config_path = get_compiler_config_path(model_name, args.output_dir)
        has_config = config_path.exists() or compiler_config_path.exists()

    if not has_config:
        print(
            f"Error: Config not found for {model_name}\n"
            f"Run discovery first:\n"
            f"  python3 -m torq.gen_config discover --model {model_path}",
            file=sys.stderr,
        )
        return 1

    extra_args = _build_extra_args(args, _RUN_FLAGS)

    # Use '_full' filter so it matches both full-model (_full_model) and
    # subgraph-full (_full) test cases.
    test_filter = "_full"

    return _run_pytest(
        test_file=test_file,
        model_path=model_path,
        test_filter=test_filter,
        output_dir=args.output_dir,
        extra_args=extra_args,
    )


def cmd_view(args: argparse.Namespace) -> int:
    """View an executor config file (report or compiler JSON)."""
    try:
        path = _resolve_config_path(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not path.exists():
        print(f"Error: Config file not found: {path}", file=sys.stderr)
        return 1

    data = load_config(path)

    # Detect compiler JSON and show a simple assignment table
    if "op_assignments" in data and "ops" not in data:
        _print_compiler_summary(data)
        return 0

    if args.layer:
        print_layer_details(data, args.layer)
    else:
        print_summary(data)
    return 0


def _print_compiler_summary(data: dict) -> None:
    """Print a summary of compiler-format JSON (op_assignments → executor)."""
    model_name = data.get("model_name", "unknown")
    assignments = data.get("op_assignments", {})

    print("=" * 60)
    print(f"MODEL: {model_name}  (compiler format)")
    print("=" * 60)
    print(f"\nTotal assignments: {len(assignments)}")

    if not assignments:
        return

    # Count by executor
    counts = {}
    for loc, entry in sorted(assignments.items()):
        exec_name = entry.get("executor", "?")
        counts[exec_name] = counts.get(exec_name, 0) + 1

    print("\nExecutor distribution:")
    for exec_name in ("nss", "css", "host"):
        if exec_name in counts:
            print(f"  {exec_name.upper()}: {counts[exec_name]}")

    print("\nAssignments:")
    for loc, entry in sorted(assignments.items()):
        exec_name = entry.get("executor", "?")
        print(f"  {loc} → {exec_name}")
    print()


def _resolve_config_path(args: argparse.Namespace) -> Path:
    """Resolve the JSON path from --model or positional config (shared by edit/view)."""
    if args.config:
        return Path(args.config)
    if args.model:
        model_name = Path(args.model).stem
        return get_config_path(model_name, args.output_dir)
    raise ValueError("Either --model or a config path must be provided.")


def _detect_compiler_json(data: dict, path: Path) -> Optional[str]:
    """Check if data looks like a compiler JSON and return an error message."""
    if "op_assignments" in data and "ops" not in data:
        expected = path.with_name(path.stem.replace("_compiler", "") + path.suffix)
        if expected == path:
            expected = path.with_name(path.stem + "_report" + path.suffix)
        return (
            f"Error: This appears to be a compiler JSON file (contains 'op_assignments').\n"
            f"       The edit command requires a report JSON file (contains 'ops').\n"
            f"       Expected: {expected}\n"
            f"       Found:    {path}"
        )
    return None


def _print_layer_list(data: dict, filter_query: str = "") -> None:
    """Print a formatted list of available layers.

    Uses the same per-layer status table format as the final report,
    showing NSS/CSS/Host status and recommended executor for each layer.
    If *filter_query* is provided, only layers whose ID contains the
    query (case-insensitive) are shown.
    """
    ops = data.get("ops", {})
    if not ops:
        print("No layers found in config.")
        return

    query = filter_query.lower()
    rows = []
    for layer_id, op_data in sorted(
        ops.items(), key=lambda x: x[1].get("_node_index", float("inf"))
    ):
        if query and query not in layer_id.lower():
            continue
        executors = op_data.get("executors", {})
        statuses = {
            "nss": executors.get("nss", {}).get("status", "-"),
            "css": executors.get("css", {}).get("status", "-"),
            "host": executors.get("host", {}).get("status", "-"),
        }
        rows.append(
            {
                "layer_id": layer_id,
                "statuses": statuses,
                "recommended": op_data.get("recommended_executor", "-"),
            }
        )

    if not rows:
        print(f"No layers match '{filter_query}'.")
        all_ids = list(ops.keys())
        print(
            f"Available layers: {', '.join(all_ids[:10])}"
            + ("..." if len(all_ids) > 10 else "")
        )
        return

    table_lines = format_per_layer_status_table(rows, use_color=sys.stdout.isatty())
    for line in table_lines:
        print(line)
    print(
        f"\nTotal: {len(rows)} layer(s)"
        + (f" (filtered from {len(ops)})" if query else "")
    )


def cmd_edit(args: argparse.Namespace) -> int:
    """Edit the recommended executor for a specific layer."""
    try:
        path = _resolve_config_path(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not path.exists():
        print(f"Error: Config file not found: {path}", file=sys.stderr)
        return 1

    data = load_config(path)

    # Guard against accidentally passing a compiler JSON
    compiler_error = _detect_compiler_json(data, path)
    if compiler_error:
        print(compiler_error, file=sys.stderr)
        return 1

    ops = data.get("ops", {})

    # --list mode: just print layers and exit
    if args.list is not None:
        _print_layer_list(data, filter_query=args.list)
        return 0

    # Determine target layer(s) from --layer
    target_layers: List[str] = []
    if args.layer:
        query = args.layer.lower()
        # Special case: "ALL" matches every layer
        if query == "all":
            target_layers = sorted(ops.keys())
        # If the query contains fnmatch wildcards, use fnmatch
        elif "*" in query or "?" in query:
            target_layers = sorted(
                [lid for lid in ops if fnmatch.fnmatch(lid.lower(), query)]
            )
        else:
            # Exact match first (case-insensitive), then substring fallback
            exact = [lid for lid in ops if lid.lower() == query]
            if exact:
                target_layers = sorted(exact)
            else:
                target_layers = sorted(
                    [lid for lid in ops if query in lid.lower()]
                )
        if not target_layers:
            print(
                f"Error: No layers match '{args.layer}'.", file=sys.stderr
            )
            available = list(ops.keys())[:10]
            print(
                f"Available layers: {', '.join(available)}"
                + ("..." if len(ops) > 10 else ""),
                file=sys.stderr,
            )
            return 1
        if len(target_layers) > 1:
            print(
                f"Batch edit: {len(target_layers)} layer(s) match '{args.layer}'"
            )
            for lid in target_layers:
                print(f"  - {lid}")
    else:
        print(
            "Error: --layer is required (or use --list to see available layers).",
            file=sys.stderr,
        )
        return 1

    # Apply edits to all target layers
    changed = False
    for layer_id in target_layers:
        # Update recommended executor
        if args.executor is not None:
            ops[layer_id]["recommended_executor"] = args.executor
            changed = True

        # Update tolerance if provided
        if args.tolerance_avg is not None or args.tolerance_max is not None:
            op_data = ops[layer_id]
            tol = op_data.get("tolerance_used", {})
            if args.tolerance_avg is not None:
                tol["fp_avg_tol"] = args.tolerance_avg
            if args.tolerance_max is not None:
                tol["fp_max_tol"] = args.tolerance_max
            op_data["tolerance_used"] = tol
            changed = True

    if changed:
        data["final_report_text"] = generate_final_report_text(data)

    if args.executor is not None:
        if len(target_layers) == 1:
            print(
                f"Updated recommended_executor for '{target_layers[0]}' to '{args.executor}'"
            )
        else:
            print(
                f"Updated recommended_executor for {len(target_layers)} layer(s) to '{args.executor}'"
            )

    if args.tolerance_avg is not None or args.tolerance_max is not None:
        if len(target_layers) == 1:
            op_data = ops[target_layers[0]]
            tol = op_data.get("tolerance_used", {})
            print(f"Updated tolerance for '{target_layers[0]}' to {tol}")
        else:
            print(f"Updated tolerance for {len(target_layers)} layer(s)")

    if changed:
        save_config(path, data)

        # Also regenerate compiler JSON so both files stay in sync
        compiler_path = path.with_name(path.stem + "_compiler" + path.suffix)
        model_name = data.get("model_name")
        compiler_data = generate_compiler_config(data, model_name)
        save_config(compiler_path, compiler_data)
        print(f"Updated compiler JSON: {compiler_path}")
    else:
        print("No changes made (use --executor or --tolerance-* to edit).")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for torq-gen-config."""
    parser = argparse.ArgumentParser(
        prog="torq-gen-config",
        description="Generate, view, edit, and run TORQ executor configurations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common_run_args(p):
        """Add arguments shared by discover and run subcommands."""
        p.add_argument("--model", required=True, help="Path to the ONNX model")
        p.add_argument(
            "--output-dir",
            help="Directory for executor config JSON (default: current directory)",
        )
        p.add_argument("--test-file", help="Path to test_onnx_gen_config.py")
        p.add_argument(
            "--auto-convert-bf16",
            action="store_true",
            help="Automatically convert FP32 ONNX models to BF16",
        )
        p.add_argument("--log-file", help="Redirect output to log file")
        p.add_argument(
            "extra_options",
            nargs="*",
            help="Extra options passed directly to pytest. Use '--' before flags starting with '-' (e.g., '-- -s -v')",
        )

    # discover
    discover_parser = subparsers.add_parser(
        "discover", help="Run executor discovery on an ONNX model"
    )
    _add_common_run_args(discover_parser)
    discover_parser.add_argument(
        "--skip-mode",
        action="store_true",
        help="Stop after first success per layer (--skip-mode)",
    )
    discover_parser.add_argument(
        "--skip-executors",
        help="Comma-separated list of executors to skip (e.g., nss,css)",
    )
    discover_parser.add_argument(
        "--save-bf16-model", help="Save converted BF16 model to path"
    )
    discover_parser.add_argument("--subgraph-from", help="Start op name for subgraph")
    discover_parser.add_argument("--subgraph-to", help="End op name for subgraph")
    discover_parser.add_argument(
        "--collect-timing", action="store_true", help="Collect runtime timing data"
    )
    discover_parser.add_argument(
        "--timing-runs", type=int, help="Number of runtime runs for timing average"
    )
    discover_parser.add_argument(
        "--recommend-by-timing",
        action="store_true",
        help="Recommend fastest executor based on timing",
    )
    discover_parser.add_argument(
        "--dedup-layers",
        action="store_true",
        help="Detect duplicate layers and copy results",
    )
    discover_parser.set_defaults(func=cmd_discover)

    # run (full model)
    run_parser = subparsers.add_parser(
        "run", help="Run full model test with discovered executor assignments"
    )
    _add_common_run_args(run_parser)
    run_parser.add_argument(
        "--debug-ir",
        default="tmp",
        help="Dump IR to directory for debugging (default: tmp)",
    )
    run_parser.add_argument(
        "--recompute-cache",
        action="store_true",
        help="Force recompute cached fixtures",
    )
    run_parser.add_argument(
        "--subgraph-from",
        help="Start op name for subgraph (output tensor name or OpType_outputName)",
    )
    run_parser.add_argument(
        "--subgraph-to",
        help="End op name for subgraph (output tensor name or OpType_outputName)",
    )
    run_parser.set_defaults(func=cmd_run)

    # view
    view_parser = subparsers.add_parser("view", help="View executor config")
    view_parser.add_argument(
        "config",
        nargs="?",
        help="Path to executor config JSON (optional; overrides --model)",
    )
    view_parser.add_argument(
        "--model", help="Path to ONNX model (auto-resolves JSON from model name)"
    )
    view_parser.add_argument(
        "--output-dir",
        help="Directory where config JSON is located (default: current directory)",
    )
    view_parser.add_argument("layer", nargs="?", help="Optional layer ID for details")
    view_parser.set_defaults(func=cmd_view)

    # edit
    edit_parser = subparsers.add_parser("edit", help="Edit executor config")
    edit_parser.add_argument(
        "config",
        nargs="?",
        help="Path to executor config JSON (optional; overrides --model)",
    )
    edit_parser.add_argument(
        "--model", help="Path to ONNX model (auto-resolves JSON from model name)"
    )
    edit_parser.add_argument(
        "--output-dir",
        help="Directory where config JSON is located (default: current directory)",
    )
    edit_parser.add_argument("--layer", help="Layer ID to edit. Supports exact name, substring, fnmatch wildcards (*, ?), or ALL for every layer.")
    edit_parser.add_argument(
        "--executor", help="Set recommended executor (nss/css/host or null)"
    )
    edit_parser.add_argument(
        "--tolerance-avg", type=float, help="Set fp_avg_tol for this layer"
    )
    edit_parser.add_argument(
        "--tolerance-max", type=float, help="Set fp_max_tol for this layer"
    )
    edit_parser.add_argument(
        "--list",
        nargs="?",
        const="",
        metavar="FILTER",
        help="List available layers and exit. Optional FILTER substring to match layer IDs.",
    )
    edit_parser.set_defaults(func=cmd_edit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
