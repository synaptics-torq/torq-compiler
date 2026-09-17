# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI entry point for torq-gen-config.

Provides subcommands to discover, view, edit, and run TORQ executor configurations.

``discover`` and ``run`` execute in-process via
:mod:`torq.gen_config._runner`; no source checkout needed.
"""

import argparse
import fnmatch
import sys
from pathlib import Path
from typing import List, Optional

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
from torq.lab.quantization.onnx import quantize_onnx_model
from torq.lab.quantization.onnx.static import (
    add_onnx_static_quantization_args,
    add_static_quant_flags,
)


def _validate_model_and_flags(args: argparse.Namespace) -> Optional[str]:
    """Shared discover/run validation; returns an error message or None."""
    model_path = args.model
    if not model_path:
        return "--model is required (or provide it via --config)"
    if not Path(model_path).exists():
        return f"Model not found: {model_path}"
    if str(model_path).lower().endswith(".tflite"):
        return "standalone discovery/run supports ONNX models only; TFLite discovery was removed"
    if args.auto_convert_bf16 and args.quantize:
        return "--auto-convert-bf16 and --quantize are mutually exclusive."
    if getattr(args, "auto_convert_int32", False) and args.quantize:
        return "--auto-convert-int32 and --quantize are mutually exclusive."
    return None


def cmd_discover(args: argparse.Namespace) -> int:
    """Run executor discovery on an ONNX model (in-process)."""
    from torq.gen_config._options import DiscoveryConfig

    # Build config from args + config files
    cfg = DiscoveryConfig.from_config_and_args(args)
    error = _validate_model_and_flags(args) if args.model else (
        "--model is required (or provide it via --config)" if not cfg.model_path else None
    )
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    from torq.gen_config._runner import run_discovery

    return run_discovery(cfg)


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full model test with discovered executor assignments."""
    from torq.gen_config._options import DiscoveryConfig

    # Build config from args + config files
    cfg = DiscoveryConfig.from_config_and_args(args)
    error = _validate_model_and_flags(args) if args.model else (
        "--model is required (or provide it via --config)" if not cfg.model_path else None
    )
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    model_path = cfg.model_path
    output_dir = cfg.output_dir

    # Verify config exists (report JSON or compiler JSON)
    model_name = Path(model_path).stem
    subgraph_mode = cfg.subgraph_mode

    if subgraph_mode:
        # For subgraph runs, accept any torq_gen_config JSON in the output dir
        search_dir = Path(output_dir) if output_dir else Path(".")
        has_config = any(search_dir.glob("torq_gen_config_*.json"))
    else:
        config_path = get_config_path(model_name, output_dir)
        compiler_config_path = get_compiler_config_path(model_name, output_dir)
        has_config = config_path.exists() or compiler_config_path.exists()

    if not has_config:
        print(
            f"Error: Config not found for {model_name}\n"
            f"Run discovery first:\n"
            f"  python3 -m torq.gen_config discover --model {model_path}",
            file=sys.stderr,
        )
        return 1

    from torq.gen_config._runner import run_full_model

    return run_full_model(cfg)


def cmd_quantize(args: argparse.Namespace) -> int:
    """Quantize an FP32 ONNX model to integer QDQ (optionally full-integer)."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else model_path.with_suffix(".int8.onnx")

    try:
        quantize_onnx_model(
            model_path,
            output_path,
            method="static",
            num_calib=args.num_calib,
            dataset=args.dataset,
            per_channel=args.per_channel,
            full_integer=args.full_integer,
            quant_format=args.quant_format,
            quant_dtype=args.quant_dtype,
        )
        if args.full_integer:
            print(f"Full-integer quantized model saved to: {output_path}")
        else:
            print(f"Quantized model saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error quantizing model: {e}", file=sys.stderr)
        return 1


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
        p.add_argument(
            "--config",
            action="append",
            default=[],
            help="Config JSON file(s) to layer for the discovery config (repeatable, later wins)",
        )
        p.add_argument(
            "--model",
            required=False,
            help="Path to the ONNX model (.onnx)",
        )
        p.add_argument(
            "--output-dir",
            help="Directory for executor config JSON (default: current directory)",
        )
        p.add_argument(
            "--torq-hw",
            default="default",
            metavar="TARGET",
            help="Torq hardware target, e.g. SL2610 or a chip name from "
            "tests/testdata/chips; default: SL2610 via 'default'",
        )
        p.add_argument(
            "--torq-hw-type",
            dest="torq_hw_type",
            default="sim",
            help="Runtime device type passed to torq-run-module --torq_hw_type "
            "(default: sim)",
        )
        p.add_argument(
            "--compiler-option",
            action="append",
            default=[],
            metavar="OPT",
            help="Extra option for torq-compile; repeatable",
        )
        p.add_argument(
            "--runtime-option",
            action="append",
            default=[],
            help="Extra option for torq-run-module; repeatable",
        )
        p.add_argument(
            "--auto-convert-bf16",
            action="store_true",
            help="Automatically convert FP32 models to BF16",
        )
        p.add_argument(
            "--auto-convert-int32",
            action="store_true",
            help="Automatically convert INT64 ONNX tensors to INT32 "
            "(applied after --auto-convert-bf16 when both are enabled)",
        )
        p.add_argument("--log-file", help="Redirect output to log file")
        p.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Show detailed logs (JSON cache activity, MLIR conversion, "
            "comparison metrics, skip reasons)",
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
        "--skip-ops",
        help="Comma-separated list of ONNX op types to skip (e.g., MaxPool,Add). "
        "Skipped layers get recommended_executor=null in the JSON.",
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
    discover_parser.add_argument(
        "--recompute-cache",
        action="store_true",
        help="Force recompute cached fixtures",
    )
    discover_parser.add_argument(
        "--debug-ir",
        default=None,
        nargs="?",
        const="tmp",
        help="Dump IR to directory for debugging (default: tmp)",
    )
    add_onnx_static_quantization_args(discover_parser)
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
    add_onnx_static_quantization_args(run_parser)
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

    # quantize
    quantize_parser = subparsers.add_parser(
        "quantize", help="Quantize an FP32 ONNX model to integer QDQ"
    )
    quantize_parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config JSON file(s) to layer for the quantization config (repeatable, later wins)",
    )
    quantize_parser.add_argument("--model", required=False, help="Path to the FP32 ONNX model")
    quantize_parser.add_argument(
        "--output", help="Output path for the quantized ONNX model (default: <model>.int8.onnx)"
    )
    quantize_parser.add_argument(
        "--num-calib",
        type=int,
        default=20,
        help="Number of synthetic calibration samples (default: 20)",
    )
    quantize_parser.add_argument(
        "--dataset",
        type=Path,
        help="Calibration dataset (not implemented yet, check back soon)",
    )
    add_static_quant_flags(quantize_parser)
    quantize_parser.set_defaults(func=cmd_quantize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
