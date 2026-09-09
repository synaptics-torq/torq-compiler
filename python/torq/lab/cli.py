# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""``torq-lab`` command line: a thin wrapper over ModelPipeline."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from torq.lab import artifact
from torq.lab.pipeline import ModelPipeline
from torq.lab.summary import Plan, StageTime, Summary, format_inspect, format_plan, format_summary
from torq.lab.types import LabError, PipelineConfig, RemoteTarget, load_config

logger = logging.getLogger("torq.lab.cli")

# The CLI entry-point name
PROG = "torq-lab"

# Exit codes.
_OK = 0
_ERROR = 1
_COMPARISON_FAILED = 2


def _hint(rest: str) -> str:
    """Format a copy-pasteable command hint, e.g. _hint('run model.vmfb')."""
    return f"{PROG} {rest}"


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


def _build_plan(command: str, config: PipelineConfig, remote: Optional[RemoteTarget]) -> Plan:
    model_path = Path(config.model_path)
    import_output = Path(config.work_dir) / f"{model_path.stem}.mlir" if model_path.suffix in (".onnx", ".tflite") else None
    if model_path.suffix in (".onnx", ".tflite", ".mlir"):
        compile_output = Path(config.vmfb_path) if config.vmfb_path else Path(config.work_dir) / f"{model_path.stem}.vmfb"
    else:
        compile_output = None
    if config.input_npy:
        input_source = f"npy files: {[str(p) for p in config.input_npy]}"
    elif config.random_inputs:
        input_source = f"random (seed={config.input_seed if config.input_seed is not None else 1234})"
    else:
        input_source = "none"
    return Plan(
        command=command,
        model_path=model_path,
        import_output=import_output,
        compile_output=compile_output,
        execution_target=f"remote:{remote.address}" if remote else config.runtime_hw_type,
        input_source=input_source,
        artifacts_dir=Path(config.work_dir),
        touches_board=remote is not None,
    )


def _run_summary(
    pipe, command: str, compile_result, run_result, status: str,
    reference: Optional[str] = None, next_command: Optional[str] = None,
) -> Summary:
    config = pipe.config
    stages = []
    if compile_result is not None:
        stages.append(StageTime("compile", compile_result.elapsed))
    if run_result is not None:
        stages.append(StageTime("run", run_result.wall_time or 0.0))
    if config.input_npy:
        input_provenance = f"npy: {[str(p) for p in config.input_npy]}"
    elif config.random_inputs:
        input_provenance = f"random (seed={config.input_seed if config.input_seed is not None else 1234})"
    else:
        input_provenance = "none"
    quality = None
    if run_result is not None:
        quality, _ = _profile_quality(pipe, run_result)
    return Summary(
        command=command,
        status=status,
        stages=stages,
        target=f"remote:{pipe.remote.address}" if pipe.remote else config.runtime_hw_type,
        input_provenance=input_provenance,
        profile_quality=quality,
        reference=reference,
        next_command=next_command or _hint(f"inspect {pipe._vmfb_path()}"),
    )


def _emit_summary(args, summary: Summary) -> None:
    if _opt(args, "json", False):
        print(json.dumps(summary.to_dict()))
    else:
        logger.info(format_summary(summary))


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


def _status_for(comparison) -> str:
    if comparison is not None and not comparison.passed:
        return "comparison-failed"
    return "ok"


def _cmd_compile(args) -> int:
    config = _config_from_args(args)
    remote = _remote_from_args(args)
    if _opt(args, "print_plan", False):
        print(format_plan(_build_plan("compile", config, remote)))
        return _OK
    pipe = ModelPipeline(config, remote)
    try:
        compile_result = pipe.compile()
    except Exception as exc:
        pipe.write_manifest(status="error", diagnostics=str(exc))
        logger.error("Compile failed: %s", exc)
        return _ERROR
    pipe.write_manifest(compile_result=compile_result)
    logger.info("Compiled: %s", compile_result.vmfb_path)
    _emit_summary(args, _run_summary(
        pipe, "compile", compile_result, None, "ok",
        next_command=_hint(f"run {compile_result.vmfb_path}"),
    ))
    return _OK


def _cmd_run(args) -> int:
    config = _config_from_args(args)
    remote = _remote_from_args(args)
    if _opt(args, "print_plan", False):
        print(format_plan(_build_plan("run", config, remote)))
        return _OK
    pipe = ModelPipeline(config, remote)
    compile_result = None
    try:
        compile_result, vmfb = pipe.ensure_vmfb()
        if compile_result is not None:
            logger.info("Compiling %s -> %s", pipe.config.model_path, vmfb)
        run_result = pipe.run(vmfb)
    except Exception as exc:
        pipe.write_manifest(compile_result=compile_result, status="error", diagnostics=str(exc))
        logger.error("Run failed: %s", exc)
        return _ERROR
    manifest_path = pipe.write_manifest(compile_result=compile_result, run_result=run_result, status="ok")
    logger.info("Ran model; manifest: %s", manifest_path)
    _emit_summary(args, _run_summary(pipe, "run", compile_result, run_result, "ok"))
    return _OK


def _cmd_verify(args) -> int:
    # nargs="+" on --golden greedily swallows a trailing positional (e.g.
    # `verify --golden a.npy b.npy model.mlir` leaves args.model unset), so
    # detect that case and point at the corrected command rather than the
    # generic "model is required" error.
    golden = list(_opt(args, "golden", []) or [])
    if not args.model and not args.config and golden:
        swallowed = [g for g in golden if Path(g).suffix in _MODEL_SUFFIXES]
        if swallowed:
            model = swallowed[-1]
            rest = [g for g in golden if g != model]
            logger.error(
                "'model' is required; --golden swallowed the trailing positional. Try: %s",
                _hint(f"verify {model} --golden {' '.join(rest)}"),
            )
            return _ERROR

    config = _config_from_args(args)
    remote = _remote_from_args(args)
    if _opt(args, "print_plan", False):
        print(format_plan(_build_plan("verify", config, remote)))
        return _OK
    if not config.expected_output_npy and not config.reference and Path(config.model_path).suffix == ".onnx":
        config.reference = f"onnx:{Path(config.model_path)}"
    if not config.expected_output_npy and not config.reference:
        logger.error(
            "no reference to verify against: pass --golden PATH [PATH ...] or --reference onnx:PATH"
        )
        return _ERROR

    pipe = ModelPipeline(config, remote)
    compile_result = None
    try:
        compile_result, vmfb = pipe.ensure_vmfb()
        if compile_result is not None:
            logger.info("Compiling %s -> %s", pipe.config.model_path, vmfb)
        run_result = pipe.run(vmfb)
        comparison = pipe.verify_outputs(run_result)
    except Exception as exc:
        pipe.write_manifest(compile_result=compile_result, status="error", diagnostics=str(exc))
        logger.error("verify failed: %s", exc)
        return _ERROR
    status = _status_for(comparison)
    manifest_path = pipe.write_manifest(
        compile_result=compile_result, run_result=run_result, comparison=comparison, status=status
    )
    logger.info("Verified; manifest: %s", manifest_path)
    reference = ", ".join(str(p) for p in config.expected_output_npy) or config.reference
    _emit_summary(args, _run_summary(pipe, "verify", compile_result, run_result, status, reference=reference))
    return _OK if status == "ok" else _COMPARISON_FAILED


def _profile_quality(pipe, run_result):
    """Classify what a profiling run actually produced, from run_result's paths.

    Returns ``(quality, detail)``: ``("annotated", viewer_path)`` when an
    annotated report or viewer exists, ``("raw", reason)`` when only the host
    CSV exists (no debug info to annotate against, or the ``[profile]`` extra
    is missing), or ``(None, None)`` when neither was produced.
    """
    if run_result.annotated_profile or run_result.perfetto_viewer:
        return "annotated", str(run_result.perfetto_viewer or run_result.annotated_profile)
    if run_result.host_profile:
        if not pipe._has_debug_info():
            return "raw", "no debug info to annotate against"
        from torq.lab import profiling

        reason = "the [profile] extra is not installed" if profiling.pd is None else "annotation did not run"
        return "raw", reason
    return None, None


def _cmd_profile(args) -> int:
    config = _config_from_args(args)
    remote = _remote_from_args(args)
    if _opt(args, "print_plan", False):
        print(format_plan(_build_plan("profile", config, remote)))
        return _OK
    if config.profiles_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        config.profiles_dir = Path(config.work_dir) / "profiles" / stamp

    pipe = ModelPipeline(config, remote)
    compile_result = None
    try:
        compile_result, run_result = pipe.profile()
    except Exception as exc:
        pipe.write_manifest(compile_result=compile_result, status="error", diagnostics=str(exc))
        logger.error("Profile failed: %s", exc)
        return _ERROR

    quality, detail = _profile_quality(pipe, run_result)
    if quality is None:
        diag = "profiling produced neither a host profile nor an annotated report"
        pipe.write_manifest(compile_result=compile_result, run_result=run_result, status="error", diagnostics=diag)
        logger.error(diag)
        return _ERROR

    manifest_path = pipe.write_manifest(compile_result=compile_result, run_result=run_result, status="ok")
    logger.info("Profiled (%s: %s); manifest: %s", quality, detail, manifest_path)
    _emit_summary(args, _run_summary(pipe, "profile", compile_result, run_result, "ok"))
    return _OK


def _cmd_inspect(args) -> int:
    model = _opt(args, "model")
    if not model and args.config:
        model = load_config([Path(c) for c in args.config]).get("model_path")
    if not model:
        raise SystemExit("error: 'model' is required unless --config supplies it")
    if not Path(model).exists():
        raise SystemExit(f"error: no such model or artifact: {model}")

    manifest_override = Path(_opt(args, "artifact")) if _opt(args, "artifact") else None
    source_override = Path(_opt(args, "source")) if _opt(args, "source") else None
    info = artifact.describe(Path(model), manifest=manifest_override, source=source_override)
    if _opt(args, "function"):
        info.function = args.function
        info.provenance["function"] = "--function override"

    annotated_possible = bool(
        info.debug_dir and Path(info.debug_dir).is_dir() and any(Path(info.debug_dir).iterdir())
    )
    target = info.vmfb_path or Path(model)
    hints = [
        _hint(f"run {target}"),
        _hint(f"verify {target} --golden GOLDEN.npy"),
        _hint(f"profile {target}"),
    ]
    if _opt(args, "json", False):
        payload = info.to_manifest_dict()
        payload["annotated_profile_possible"] = annotated_possible
        payload["hints"] = hints
        print(json.dumps(payload))
    else:
        print(format_inspect(info, hints, annotated_possible))
    return _OK


def _build_parser() -> argparse.ArgumentParser:
    epilog = f"""
Delegated command groups (see `{PROG} <group> --help`):
  gen_config   Per-layer NSS/CSS/Host executor discovery (discover/run/view/edit)
  quantize     Static int8 ONNX quantization
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
    compile_sub.set_defaults(handler=_cmd_compile)

    # run command
    run_sub = subparsers.add_parser("run", help="Compile if needed, then run a model", allow_abbrev=False)
    _add_model_arg(run_sub, ".onnx/.tflite/.mlir (compiled first) or .vmfb to run (optional when --config supplies it)")
    _add_shared_options(run_sub)
    _add_target_options(run_sub)
    _add_compile_options(run_sub)
    _add_run_options(run_sub)
    _add_remote_options(run_sub)
    _add_output_options(run_sub)
    run_sub.set_defaults(handler=_cmd_run)

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
    verify_sub.set_defaults(handler=_cmd_verify)

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
    profile_sub.set_defaults(handler=_cmd_profile)

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
    inspect_sub.set_defaults(handler=_cmd_inspect)

    return parser


def main(argv=None) -> int:
    # Intercept delegated commands (gen_config, quantize) before building the parser
    # to avoid importing onnxruntime at parse time.
    _DELEGATED = {"gen_config", "quantize"}
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _DELEGATED:
        from torq.gen_config.cli import main as gen_config_main
        # `quantize` is a torq-gen-config subcommand already; pass it through
        # as-is. `gen_config X ...` maps to torq-gen-config's `X ...`.
        forwarded = argv if argv[0] == "quantize" else argv[1:]
        return gen_config_main(forwarded)

    if not argv:
        from torq.lab.interactive import run_interactive
        return run_interactive()

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return args.handler(args)
    except LabError as exc:
        # Config loading and other pre-pipeline steps run outside a handler's
        # own try/except; surface their errors cleanly rather than as a traceback.
        logger.error("%s", exc)
        return _ERROR
