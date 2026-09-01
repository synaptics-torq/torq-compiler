# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""``torq-lab`` command line: a thin wrapper over ModelPipeline."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from torq.lab.pipeline import ModelPipeline
from torq.lab.types import PipelineConfig, RemoteTarget, load_config

logger = logging.getLogger("torq.lab.cli")

# Exit codes.
_OK = 0
_ERROR = 1
_COMPARISON_FAILED = 2


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", nargs="?", help="Input model: .mlir for compile/compile-run, .vmfb for run (optional when --config supplies it)")
    parser.add_argument("--config", action="append", default=[], help="Config JSON file(s) to layer for the pipeline config (repeatable, later wins); an alternative to the individual flags")
    parser.add_argument("--work-dir", help="Artifact directory (default: ./<model-stem>-run)")
    parser.add_argument("--chip", default="SL2610", help="Chip target for --torq-hw")
    parser.add_argument("--runtime-hw-type", default="sim", help="Runtime HW type (sim, aws_fpga, astra_machina, ...)")
    parser.add_argument("--function", help="Entry function name (default: parsed from MLIR, else 'main')")
    parser.add_argument("--compiler-option", action="append", default=[], help="Extra torq-compile flag (repeatable)")
    parser.add_argument("--runtime-option", action="append", default=[], help="Extra torq-run-module flag (repeatable)")
    parser.add_argument("--dump-ir", action="store_true", help="Dump IR after each pass into <work-dir>/debug/ir")
    parser.add_argument("--dump-phases", action="store_true", help="Dump compilation phases into <work-dir>/phases")
    parser.add_argument("--profile-compile", action="store_true", help="Enable compile-time profiling")
    parser.add_argument("--profile-runtime", action="store_true", help="Enable runtime host profiling")
    parser.add_argument("--convert-io-dtypes", nargs="+", default=[], metavar="all|input:IDX|output:IDX|!input:IDX|!output:IDX", help="VMFB uses converted public I/O dtypes; must match its compilation")
    parser.add_argument("--input-npy", action="append", default=[], help="Input .npy file (repeatable)")
    parser.add_argument("--expected-output-npy", action="append", default=[], help="Expected output .npy for comparison (repeatable)")
    parser.add_argument("--random-inputs", action="store_true", help="Generate random inputs from the MLIR spec")
    parser.add_argument("--remote", help="Remote board: adb, an adb serial, user@host, or a hostname/IP for SSH")
    parser.add_argument("--remote-port", type=int, default=22, help="SSH port")
    parser.add_argument("--remote-key", help="SSH private key path")
    parser.add_argument("--remote-runner", help="Absolute path of torq-run-module on the device")
    parser.add_argument("--stage-runner", help="Local torq-run-module to stage onto the device")
    parser.add_argument("--timeout", type=int, help="Per-tool timeout in seconds")


def _config_from_args(args) -> PipelineConfig:
    if args.config:
        merged = load_config([Path(c) for c in args.config])
        # Explicit positional/flag values override the layered config files.
        if args.model:
            merged["model_path"] = args.model
        if args.work_dir:
            merged["work_dir"] = args.work_dir
        if getattr(args, "output", None):
            merged["vmfb_path"] = args.output
        return PipelineConfig.from_dict(merged)
    if not args.model:
        raise SystemExit("error: 'model' is required unless --config is given")
    work_dir = Path(args.work_dir) if args.work_dir else Path.cwd() / f"{Path(args.model).stem}-run"
    return PipelineConfig(
        model_path=Path(args.model),
        work_dir=work_dir,
        chip=args.chip,
        runtime_hw_type=args.runtime_hw_type,
        function=args.function,
        compiler_options=list(args.compiler_option),
        runtime_options=list(args.runtime_option),
        dump_ir=args.dump_ir,
        dump_phases=args.dump_phases,
        profile_compile=args.profile_compile,
        profile_runtime=args.profile_runtime,
        convert_io_dtypes=list(args.convert_io_dtypes),
        random_inputs=args.random_inputs,
        input_npy=[Path(p) for p in args.input_npy],
        expected_output_npy=[Path(p) for p in args.expected_output_npy],
        timeout=args.timeout,
        vmfb_path=Path(args.output) if getattr(args, "output", None) else None,
    )


def _remote_from_args(args) -> Optional[RemoteTarget]:
    if args.config:
        remote = load_config([Path(c) for c in args.config]).get("remote")
        if remote:
            # --remote, if given, overrides the address from the config files.
            if args.remote:
                remote = {**remote, "address": args.remote}
            return RemoteTarget.from_dict(remote)
    if not args.remote:
        return None
    return RemoteTarget(
        address=args.remote,
        port=args.remote_port,
        private_key=args.remote_key,
        remote_runner_path=args.remote_runner,
        stage_runner=args.stage_runner,
    )


def _status_for(comparison) -> str:
    if comparison is not None and not comparison.passed:
        return "comparison-failed"
    return "ok"


def _cmd_compile(args) -> int:
    pipe = ModelPipeline(_config_from_args(args), _remote_from_args(args))
    try:
        compile_result = pipe.compile()
    except Exception as exc:
        pipe.write_manifest(status="error", diagnostics=str(exc))
        logger.error("Compile failed: %s", exc)
        return _ERROR
    pipe.write_manifest(compile_result=compile_result)
    logger.info("Compiled: %s", compile_result.vmfb_path)
    return _OK


def _cmd_run(args) -> int:
    config = _config_from_args(args)
    if config.model_path.suffix != ".vmfb":
        logger.error(
            "run requires a .vmfb model, got %s. Use: torq-lab run %s [options]",
            config.model_path,
            config.model_path.with_suffix(".vmfb"),
        )
        return _ERROR
    pipe = ModelPipeline(config, _remote_from_args(args))
    try:
        run_result = pipe.run()
        comparison = pipe.compare(run_result)
    except Exception as exc:
        pipe.write_manifest(status="error", diagnostics=str(exc))
        logger.error("Run failed: %s", exc)
        return _ERROR
    status = _status_for(comparison)
    manifest_path = pipe.write_manifest(run_result=run_result, comparison=comparison, status=status)
    logger.info("Ran model; manifest: %s", manifest_path)
    return _OK if status == "ok" else _COMPARISON_FAILED


def _cmd_compile_run(args) -> int:
    pipe = ModelPipeline(_config_from_args(args), _remote_from_args(args))
    compile_result = None
    try:
        compile_result = pipe.compile()
        run_result = pipe.run(compile_result.vmfb_path)
        comparison = pipe.compare(run_result)
    except Exception as exc:
        pipe.write_manifest(compile_result=compile_result, status="error", diagnostics=str(exc))
        logger.error("compile-run failed: %s", exc)
        return _ERROR
    status = _status_for(comparison)
    manifest_path = pipe.write_manifest(
        compile_result=compile_result, run_result=run_result, comparison=comparison, status=status
    )
    logger.info("Compiled and ran; manifest: %s", manifest_path)
    return _OK if status == "ok" else _COMPARISON_FAILED


def _cmd_compare(args) -> int:
    # `compare` is compile-run with an output check required: it insists on
    # expected outputs and reports comparison-failed (exit 2) on mismatch. The
    # expected outputs may come from --expected-output-npy or a --config file,
    # so gate on the resolved config rather than the raw CLI flag.
    if not _config_from_args(args).expected_output_npy:
        logger.error("compare requires at least one expected output (--expected-output-npy or a config)")
        return _ERROR
    return _cmd_compile_run(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torq-lab", description="Compile, run, and profile MLIR/VMFB models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = [
        ("compile", _cmd_compile, "Compile an MLIR model to a VMFB"),
        ("run", _cmd_run, "Run a compiled VMFB"),
        ("compile-run", _cmd_compile_run, "Compile an MLIR model and run it"),
        ("compare", _cmd_compare, "Compile, run, and check outputs against --expected-output-npy"),
    ]
    for name, handler, help_text in commands:
        sub = subparsers.add_parser(name, help=help_text)
        _add_common_options(sub)
        if name == "compile":
            sub.add_argument("-o", "--output", help="Output VMFB path (default: <work-dir>/model.vmfb)")
        sub.set_defaults(handler=handler)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.handler(args)
