# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command dispatch and the installed ``torq-lab`` CLI ``main()``."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from torq.lab.pipeline import artifacts
from torq.lab.cli.output import Plan, StageTime, Summary, format_inspect, format_plan, format_summary
from torq.lab.cli.parser import PROG, _MODEL_SUFFIXES, _build_parser, _config_from_args, _opt, _remote_from_args
from torq.lab import LabError
from torq.lab.pipeline.remote import RemoteTarget
from torq.lab.pipeline.workflow import ModelPipeline, PipelineConfig, load_config

logger = logging.getLogger("torq.lab.cli")

# Exit codes.
_OK = 0
_ERROR = 1
_COMPARISON_FAILED = 2


def _hint(rest: str) -> str:
    """Format a copy-pasteable command hint, e.g. _hint('run model.vmfb')."""
    return f"{PROG} {rest}"


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
        from torq.lab.profiling import annotate

        reason = "the [profile] extra is not installed" if annotate.pd is None else "annotation did not run"
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
    info = artifacts.describe(Path(model), manifest=manifest_override, source=source_override)
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


def main(argv=None) -> int:
    # Intercept delegated commands (gen_config, quantize, analyze, convert_dtype, convert_static)
    # before building the parser to avoid importing onnxruntime/tensorflow at parse time.
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        from torq.lab.cli.interactive import run_interactive
        return run_interactive()

    task = argv[0]
    if task == "gen_config":
        from torq.gen_config.cli import main as gen_config_main

        return gen_config_main(argv[1:])
    elif task == "quantize":
        from torq.lab.quantization.onnx.cli import main as quantize_main

        return quantize_main(argv[1:], prog="torq-lab quantize")
    elif task == "analyze":
        from torq.lab.quantization.onnx.cli import analyze_main

        return analyze_main(argv[1:])
    elif task == "convert_dtype":
        from torq.lab.model_tools.dtype_conversion.onnx import main as convert_dtype_main

        return convert_dtype_main(argv[1:])
    elif task == "convert_static":
        from torq.lab.model_tools.shape_conversion.tflite import main as convert_static_main

        return convert_static_main(argv[1:])

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
