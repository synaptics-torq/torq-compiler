# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Parser construction and CLI-to-domain translation for ``torq-lab``."""

import argparse
from pathlib import Path
from typing import Optional

from torq.lab.pipeline.remote import RemoteTarget
from torq.lab.pipeline.workflow import PipelineConfig, load_config

# The CLI entry-point name
PROG = "torq-lab"


def _opt(args, name: str, default=None):
    """Get an optional argument, returning default if not present."""
    return getattr(args, name, default)


def _add_model_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("model", nargs="?", help=help_text)


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("common")
    group.add_argument("--config", action="append", default=[], help="Config JSON file(s) to layer for the pipeline config (repeatable, later wins); an alternative to the individual flags")
    group.add_argument("--work-dir", help="Artifact directory (default: ./<model-stem>-run)")
    group.add_argument("--timeout", type=int, help="Per-tool timeout in seconds")
    group.add_argument("--reuse-work-dir", action="store_true", help="Reuse a populated work directory instead of allocating a run ID")


def _add_target_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("target")
    group.add_argument("--runtime-hw-type", default=None, help="Runtime HW type (sim, aws_fpga, astra_machina, ...); default: sim")


def _add_compile_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("compile stage")
    group.add_argument("--chip", default=None, help="Chip target for --torq-hw (default: SL2610)")
    group.add_argument("--compiler-option", action="append", default=[], help="Extra torq-compile flag (repeatable)")
    group.add_argument("--dump-ir", action="store_true", help="Dump IR after each pass into <work-dir>/debug/ir")
    group.add_argument("--dump-phases", action="store_true", help="Dump compilation phases into <work-dir>/phases")
    group.add_argument("--profile-compile", action="store_true", help="Enable compile-time profiling")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("run stage")
    group.add_argument("--function", help="Entry function name (default: parsed from MLIR, else 'main')")
    group.add_argument("--runtime-option", action="append", default=[], help="Extra torq-run-module flag (repeatable)")
    group.add_argument("--convert-io-dtypes", nargs="+", default=[], metavar="all|input:IDX|output:IDX|!input:IDX|!output:IDX", help="VMFB uses converted public I/O dtypes; must match its compilation")
    group.add_argument("--input-npy", action="append", default=[], help="Input .npy file (repeatable)")
    group.add_argument("--input-spec", action="append", default=[], metavar="SPEC", help="Input tensor spec (e.g., 1x1x64xbf16; repeatable, for a VMFB with no source)")
    group.add_argument("--output-spec", action="append", default=[], metavar="SPEC", help="Output tensor spec (e.g., 1x1x64xbf16; repeatable, for a VMFB with no source)")
    group.add_argument("--seed", type=int, help="Random seed for input generation (default: 1234)")
    group.add_argument("--random-inputs", action="store_true", help="Generate random inputs from the MLIR spec")


_MODEL_SUFFIXES = (".onnx", ".tflite", ".mlir", ".vmfb")


def _add_verify_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("verification")
    group.add_argument("--golden", nargs="+", metavar="GOLDEN", default=[], help="Expected output .npy file(s) to compare against (repeatable)")
    group.add_argument("--reference", metavar="PROVIDER", help="Golden reference provider: onnx:PATH, or onnx to use the model's own ONNX source. Default: the model itself when it is .onnx and no --golden is given")


def _add_remote_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("remote target")
    group.add_argument("--remote", help="Remote board: adb, an adb serial, user@host, or a hostname/IP for SSH")
    group.add_argument("--remote-port", type=int, default=22, help="SSH port")
    group.add_argument("--remote-key", help="SSH private key path")
    group.add_argument("--remote-runner", help="Absolute path of torq-run-module on the device")
    group.add_argument("--stage-runner", help="Local torq-run-module to stage onto the device")


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument("--print-plan", action="store_true", help="Print the execution plan and exit without importing, compiling, running, or connecting")
    group.add_argument("--json", action="store_true", help="Print the summary as structured JSON instead of human text")


def _cli_overrides(args) -> dict:
    """PipelineConfig keys the user *explicitly* set on the command line.

    Used to layer CLI flags over a ``--config`` file (decision 12: CLI flags
    win). Detection: a value flag is "set" when non-``None`` (``--chip`` and
    ``--runtime-hw-type`` default to ``None`` for this reason), a ``store_true``
    flag when truthy (it can only become ``True`` by being passed), and an
    ``append`` list when non-empty.
    """
    overrides: dict = {}
    values = {
        "chip": "chip",
        "runtime_hw_type": "runtime_hw_type",
        "function": "function",
        "timeout": "timeout",
        "reference": "reference",
        "seed": "input_seed",
    }
    for arg_name, cfg_key in values.items():
        if _opt(args, arg_name) is not None:
            overrides[cfg_key] = _opt(args, arg_name)
    for flag in ("dump_ir", "dump_phases", "profile_compile", "random_inputs", "reuse_work_dir"):
        if _opt(args, flag, False):
            overrides[flag] = True
    lists = {
        "compiler_option": "compiler_options",
        "runtime_option": "runtime_options",
        "input_npy": "input_npy",
        "golden": "expected_output_npy",
        "convert_io_dtypes": "convert_io_dtypes",
        "input_spec": "input_specs",
        "output_spec": "output_specs",
    }
    for arg_name, cfg_key in lists.items():
        value = _opt(args, arg_name)
        if value:
            overrides[cfg_key] = [str(v) for v in value]
    return overrides


def _config_from_args(args) -> PipelineConfig:
    if args.config:
        merged = load_config([Path(c) for c in args.config])
        # Explicit positional/flag values override the layered config files.
        if args.model:
            merged["model_path"] = args.model
        if _opt(args, "work_dir"):
            merged["work_dir"] = _opt(args, "work_dir")
        if _opt(args, "output"):
            merged["vmfb_path"] = _opt(args, "output")
        merged.update(_cli_overrides(args))
        return PipelineConfig.from_dict(merged)
    if not args.model:
        raise SystemExit("error: 'model' is required unless --config is given")
    work_dir = Path(_opt(args, "work_dir")) if _opt(args, "work_dir") else Path.cwd() / f"{Path(args.model).stem}-run"
    return PipelineConfig(
        model_path=Path(args.model),
        work_dir=work_dir,
        chip=_opt(args, "chip") or "SL2610",
        runtime_hw_type=_opt(args, "runtime_hw_type") or "sim",
        function=_opt(args, "function"),
        compiler_options=list(_opt(args, "compiler_option", [])),
        runtime_options=list(_opt(args, "runtime_option", [])),
        dump_ir=_opt(args, "dump_ir", False),
        dump_phases=_opt(args, "dump_phases", False),
        profile_compile=_opt(args, "profile_compile", False),
        profile_runtime=_opt(args, "profile_runtime", False),
        convert_io_dtypes=list(_opt(args, "convert_io_dtypes", [])),
        random_inputs=_opt(args, "random_inputs", False),
        input_npy=[Path(p) for p in _opt(args, "input_npy", [])],
        expected_output_npy=[Path(p) for p in _opt(args, "golden", [])],
        reference=_opt(args, "reference"),
        timeout=_opt(args, "timeout"),
        vmfb_path=Path(_opt(args, "output")) if _opt(args, "output") else None,
        input_specs=list(_opt(args, "input_spec", [])),
        output_specs=list(_opt(args, "output_spec", [])),
        input_seed=_opt(args, "seed"),
        reuse_work_dir=_opt(args, "reuse_work_dir", False),
    )


def _remote_from_args(args) -> Optional[RemoteTarget]:
    if args.config:
        remote = load_config([Path(c) for c in args.config]).get("remote")
        if remote:
            # --remote, if given, overrides the address from the config files.
            if _opt(args, "remote"):
                remote = {**remote, "address": _opt(args, "remote")}
            return RemoteTarget.from_dict(remote)
    if not _opt(args, "remote"):
        return None
    return RemoteTarget(
        address=_opt(args, "remote"),
        port=_opt(args, "remote_port", 22),
        private_key=_opt(args, "remote_key"),
        remote_runner_path=_opt(args, "remote_runner"),
        stage_runner=_opt(args, "stage_runner"),
    )


def _build_parser() -> argparse.ArgumentParser:
    from torq.lab.cli import commands

    epilog = f"""
Delegated command groups (see `{PROG} <group> --help`):
  gen_config    Per-layer NSS/CSS/Host executor discovery (discover/run/view/edit)
  quantize      ONNX quantization: static (int8), dynamic (int8), or weights (int4/int8/bf16)
  analyze       ONNX quantization sensitivity analysis (per-node / per-layer reports)
  convert_dtype Convert model to Torq compatible dtypes
  convert_static Convert a dynamic TFLite model to static (default) shapes
"""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Compile, run, and profile MLIR/VMFB models.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compile command
    compile_sub = subparsers.add_parser(
        "compile",
        help="Compile an MLIR model to a VMFB",
        description="Build a deliverable artifact (VMFB + debug info + manifest) for a later run elsewhere. Emits debug info and a manifest so a later run/profile needs no compiler flags; use torq-compile if you only want a VMFB.",
        allow_abbrev=False,
    )
    _add_model_arg(compile_sub, "MLIR source to compile (optional when --config supplies it)")
    _add_shared_options(compile_sub)
    _add_target_options(compile_sub)
    _add_compile_options(compile_sub)
    compile_sub.add_argument("-o", "--output", help="Output VMFB path (default: <work-dir>/<model-stem>.vmfb)")
    _add_output_options(compile_sub)
    compile_sub.set_defaults(handler=commands._cmd_compile)

    # run command
    run_sub = subparsers.add_parser("run", help="Compile if needed, then run a model", allow_abbrev=False)
    _add_model_arg(run_sub, ".onnx/.tflite/.mlir (compiled first) or .vmfb to run (optional when --config supplies it)")
    _add_shared_options(run_sub)
    _add_target_options(run_sub)
    _add_compile_options(run_sub)
    _add_run_options(run_sub)
    _add_remote_options(run_sub)
    _add_output_options(run_sub)
    run_sub.set_defaults(handler=commands._cmd_run)

    # verify command
    verify_sub = subparsers.add_parser(
        "verify", help="Compile/run if needed, then check outputs against a reference", allow_abbrev=False
    )
    _add_model_arg(
        verify_sub,
        ".onnx/.tflite/.mlir (compiled first) or .vmfb to verify (optional when --config supplies it)",
    )
    _add_shared_options(verify_sub)
    _add_target_options(verify_sub)
    _add_compile_options(verify_sub)
    _add_run_options(verify_sub)
    _add_verify_options(verify_sub)
    _add_remote_options(verify_sub)
    _add_output_options(verify_sub)
    verify_sub.set_defaults(handler=commands._cmd_verify)

    # profile command
    profile_sub = subparsers.add_parser(
        "profile", help="Compile/run if needed with host profiling on, and report the profile", allow_abbrev=False
    )
    _add_model_arg(
        profile_sub,
        ".onnx/.tflite/.mlir (compiled first) or .vmfb to profile (optional when --config supplies it)",
    )
    _add_shared_options(profile_sub)
    _add_target_options(profile_sub)
    _add_compile_options(profile_sub)
    _add_run_options(profile_sub)
    _add_remote_options(profile_sub)
    _add_output_options(profile_sub)
    profile_sub.set_defaults(handler=commands._cmd_profile)

    # inspect command
    inspect_sub = subparsers.add_parser(
        "inspect",
        help="Describe a .vmfb/.mlir/.onnx/artifact dir and print next commands",
        description="Describe what an artifact is and what can be done with it. No compile, run, or board access.",
        allow_abbrev=False,
    )
    _add_model_arg(inspect_sub, ".vmfb, .mlir, .onnx, or an artifact directory to describe (optional when --config supplies it)")
    _add_shared_options(inspect_sub)
    inspect_sub.add_argument("--artifact", metavar="MANIFEST", help="Explicit manifest.json to resolve facts from")
    inspect_sub.add_argument("--source", metavar="MODEL", help="Explicit .mlir/.onnx source to resolve I/O from")
    inspect_sub.add_argument("--function", metavar="NAME", help="Override the resolved entry function name")
    _add_output_options(inspect_sub)
    inspect_sub.set_defaults(handler=commands._cmd_inspect)

    return parser
