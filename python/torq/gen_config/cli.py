# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI entry point for torq-gen-config.

Provides subcommands to discover, view, edit, and run TORQ executor configurations.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from torq.gen_config.core import (
    get_config_path,
    load_config,
    save_config,
)
from torq.gen_config.view import print_layer_details, print_summary


DEFAULT_TEST_FILE = "tests/test_onnx_gen_config.py"


def _find_project_root() -> Path:
    """Find the torq-compiler-dev project root by looking for tests/."""
    cwd = Path.cwd()
    for path in [cwd] + list(cwd.parents):
        if (path / "tests" / "test_onnx_gen_config.py").exists():
            return path
    return cwd


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
        cmd.append(f"--gen-config-output={output_dir}")
    cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def cmd_discover(args: argparse.Namespace) -> int:
    """Run executor discovery on an ONNX model."""
    project_root = _find_project_root()
    test_file = args.test_file or str(project_root / DEFAULT_TEST_FILE)
    model_path = args.model

    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        return 1

    extra_args: List[str] = ["--recompute-cache"]
    if args.skip_mode:
        extra_args.append("--executor-skip-mode")
    if args.skip_executors:
        extra_args.append(f"--skip-executors={args.skip_executors}")
    if args.auto_convert_bf16:
        extra_args.append("--auto-convert-bf16")
    if args.save_bf16_model:
        extra_args.append(f"--save-bf16-model={args.save_bf16_model}")
    if args.subgraph_from:
        extra_args.append(f"--subgraph-from={args.subgraph_from}")
    if args.subgraph_to:
        extra_args.append(f"--subgraph-to={args.subgraph_to}")
    if args.collect_timing:
        extra_args.append("--collect-timing")
    if args.timing_runs is not None:
        extra_args.append(f"--timing-runs={args.timing_runs}")
    if args.recommend_by_timing:
        extra_args.append("--recommend-by-timing")
    if args.dedup_layers:
        extra_args.append("--dedup-layers")
    if args.log_file:
        extra_args.append(f"--gen-config-log-file={args.log_file}")
    if args.extra_options:
        extra_args.extend(args.extra_options)

    return _run_pytest(
        test_file=test_file,
        model_path=model_path,
        test_filter="_layer_",
        output_dir=args.output_dir,
        extra_args=extra_args,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full model test with discovered executor assignments."""
    project_root = _find_project_root()
    test_file = args.test_file or str(project_root / DEFAULT_TEST_FILE)
    model_path = args.model

    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        return 1

    # Verify config exists
    model_name = Path(model_path).stem
    config_path = get_config_path(model_name, args.output_dir)
    if not config_path.exists():
        print(
            f"Error: Config not found: {config_path}\n"
            f"Run discovery first:\n"
            f"  python3 -m torq.gen_config discover --model {model_path}",
            file=sys.stderr,
        )
        return 1

    extra_args: List[str] = []
    if args.auto_convert_bf16:
        extra_args.append("--auto-convert-bf16")
    if args.debug_ir:
        extra_args.append(f"--debug-ir={args.debug_ir}")
    if args.recompute_cache:
        extra_args.append("--recompute-cache")
    if args.log_file:
        extra_args.append(f"--gen-config-log-file={args.log_file}")
    if args.extra_options:
        extra_args.extend(args.extra_options)

    return _run_pytest(
        test_file=test_file,
        model_path=model_path,
        test_filter="_full_model",
        output_dir=args.output_dir,
        extra_args=extra_args,
    )


def cmd_view(args: argparse.Namespace) -> int:
    """View an executor config file."""
    path = Path(args.config)
    if not path.exists():
        print(f"Error: Config file not found: {path}", file=sys.stderr)
        return 1

    data = load_config(path)
    if args.layer:
        print_layer_details(data, args.layer)
    else:
        print_summary(data)
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    """Edit the recommended executor for a specific layer."""
    path = Path(args.config)
    if not path.exists():
        print(f"Error: Config file not found: {path}", file=sys.stderr)
        return 1

    data = load_config(path)
    ops = data.get("ops", {})
    layer_id = args.layer

    if layer_id not in ops:
        print(f"Error: Layer '{layer_id}' not found in config.", file=sys.stderr)
        available = list(ops.keys())[:10]
        print(f"Available layers: {', '.join(available)}...", file=sys.stderr)
        return 1

    # Update recommended executor
    if args.executor is not None:
        ops[layer_id]["recommended_executor"] = args.executor
        print(f"Updated recommended_executor for '{layer_id}' to '{args.executor}'")

    # Update tolerance if provided
    if args.tolerance_avg is not None or args.tolerance_max is not None:
        op_data = ops[layer_id]
        tol = op_data.get("tolerance_used", {})
        if args.tolerance_avg is not None:
            tol["fp_avg_tol"] = args.tolerance_avg
        if args.tolerance_max is not None:
            tol["fp_max_tol"] = args.tolerance_max
        op_data["tolerance_used"] = tol
        print(f"Updated tolerance for '{layer_id}' to {tol}")

    save_config(path, data)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for torq-gen-config."""
    parser = argparse.ArgumentParser(
        prog="torq-gen-config",
        description="Generate, view, edit, and run TORQ executor configurations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # discover
    discover_parser = subparsers.add_parser(
        "discover", help="Run executor discovery on an ONNX model"
    )
    discover_parser.add_argument(
        "--model", required=True, help="Path to the ONNX model"
    )
    discover_parser.add_argument(
        "--output-dir",
        help="Directory to save the generated JSON (default: current directory)",
    )
    discover_parser.add_argument(
        "--test-file", help="Path to test_onnx_gen_config.py"
    )
    discover_parser.add_argument(
        "--skip-mode",
        action="store_true",
        help="Stop after first success per layer (--executor-skip-mode)",
    )
    discover_parser.add_argument(
        "--skip-executors",
        help="Comma-separated list of executors to skip (e.g., nss,css)",
    )
    discover_parser.add_argument(
        "--auto-convert-bf16",
        action="store_true",
        help="Automatically convert FP32 ONNX models to BF16",
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
        "--log-file", help="Redirect discovery output to log file"
    )
    discover_parser.add_argument(
        "extra_options",
        nargs="*",
        help="Extra options passed directly to pytest",
    )
    discover_parser.set_defaults(func=cmd_discover)

    # run (full model)
    run_parser = subparsers.add_parser(
        "run", help="Run full model test with discovered executor assignments"
    )
    run_parser.add_argument("--model", required=True, help="Path to the ONNX model")
    run_parser.add_argument(
        "--output-dir",
        help="Directory where executor config JSON is located (default: current directory)",
    )
    run_parser.add_argument(
        "--test-file", help="Path to test_onnx_gen_config.py"
    )
    run_parser.add_argument(
        "--auto-convert-bf16",
        action="store_true",
        help="Automatically convert FP32 ONNX models to BF16",
    )
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
        "--log-file", help="Redirect output to log file"
    )
    run_parser.add_argument(
        "extra_options",
        nargs="*",
        help="Extra options passed directly to pytest",
    )
    run_parser.set_defaults(func=cmd_run)

    # view
    view_parser = subparsers.add_parser("view", help="View executor config")
    view_parser.add_argument("config", help="Path to executor config JSON")
    view_parser.add_argument("layer", nargs="?", help="Optional layer ID for details")
    view_parser.set_defaults(func=cmd_view)

    # edit
    edit_parser = subparsers.add_parser("edit", help="Edit executor config")
    edit_parser.add_argument("config", help="Path to executor config JSON")
    edit_parser.add_argument("--layer", required=True, help="Layer ID to edit")
    edit_parser.add_argument(
        "--executor", help="Set recommended executor (nss/css/host or null)"
    )
    edit_parser.add_argument(
        "--tolerance-avg", type=float, help="Set fp_avg_tol for this layer"
    )
    edit_parser.add_argument(
        "--tolerance-max", type=float, help="Set fp_max_tol for this layer"
    )
    edit_parser.set_defaults(func=cmd_edit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
