# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""No-argument interactive mode: prompt, compile, then loop a menu over the library.

This is a prompt-and-formatting layer over :class:`~torq.lab.pipeline.workflow.ModelPipeline`
and the artifact/summary machinery -- it adds no capability of its own. All I/O goes
through the injected ``reader``/``writer`` callables so the loop is testable from
scripted input.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from torq.lab.cli.commands import _hint, _profile_quality, _run_summary, _status_for
from torq.lab.cli.output import format_inspect, format_summary
from torq.lab.pipeline.remote import RemoteTarget
from torq.lab.pipeline.workflow import ModelPipeline, PipelineConfig

_MODEL_SUFFIXES = (".onnx", ".tflite", ".mlir", ".vmfb")
_MODEL_PROMPT = "Model file (.onnx/.tflite/.mlir/.vmfb): "
_MENU_ORDER = ["run", "profile", "verify", "inspect", "gen_config", "quantize", "recompile", "quit"]
_GEN_CONFIG_ACTIONS = ["discover", "run", "view", "edit"]

_OK = 0


def run_interactive(reader=input, writer=print) -> int:
    """No-argument entry point: prompt for a model, compile it, then loop a menu.

    ``reader(prompt) -> str`` and ``writer(*args) -> None`` default to the
    builtins so production uses the terminal; tests pass a scripted reader (a
    list-backed callable) and a capturing writer. Returns a process exit code.
    """
    _try_enable_readline()
    writer("torq-lab interactive mode: press Ctrl-D to quit at any time.")
    try:
        while True:
            model_path = _prompt_model(reader, writer)
            if model_path is None:
                return _OK
            pipe = _initial_compile(model_path, reader, writer)
            if pipe is None:
                continue
            return _menu_loop(pipe, reader, writer)
    except (EOFError, KeyboardInterrupt):
        writer("")
        return _OK


def _try_enable_readline() -> None:
    try:
        import readline  # noqa: F401
    except ImportError:
        pass


def _prompt_model(reader, writer) -> Optional[Path]:
    while True:
        raw = reader(_MODEL_PROMPT).strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.exists():
            writer(f"No such file: {path}")
            continue
        if path.suffix not in _MODEL_SUFFIXES:
            writer(f"Unsupported model file; accepted kinds: {', '.join(_MODEL_SUFFIXES)}")
            continue
        return path


def _prompt_work_dir(model_path: Path, reader, writer) -> Path:
    default = Path.cwd() / f"{model_path.stem}-run"
    raw = reader(f"Work dir [{default}]: ").strip()
    return Path(raw) if raw else default


def _prompt_compile_flags(reader, writer):
    chip = reader("Chip [SL2610]: ").strip() or "SL2610"
    runtime_hw_type = reader("Runtime HW type [sim]: ").strip() or "sim"
    raw_opts = reader("Extra --compiler-option flags [none]: ").strip()
    compiler_options = raw_opts.split() if raw_opts else []
    remote_address = reader("Remote address [none = local]: ").strip() or None
    return chip, runtime_hw_type, compiler_options, remote_address


def _target_label(pipe: ModelPipeline) -> str:
    return f"remote:{pipe.remote.address}" if pipe.remote else pipe.config.runtime_hw_type


def _print_artifact_info(info, writer, header: str) -> None:
    writer(header)
    writer(f"Function: {info.function}")
    if info.io_spec:
        writer(f"Inputs: {[t.to_arg() for t in info.io_spec.inputs]}")
        writer(f"Outputs: {[t.to_arg() for t in info.io_spec.outputs]}")
    target = info.vmfb_path or info.source_path
    if target is not None:
        writer(f"Next: {_hint(f'run {target}')}")


def _compile_with_flags(model_path: Path, work_dir: Path, reader, writer) -> Optional[ModelPipeline]:
    """Compile ``model_path`` into ``work_dir``, prompting for compile flags (skipped for a .vmfb)."""
    if model_path.suffix == ".vmfb":
        config = PipelineConfig(model_path=model_path, work_dir=work_dir)
        pipe = ModelPipeline(config, None)
        info = pipe._resolve_artifact_info()
        _print_artifact_info(info, writer, header=f"VMFB: {model_path}")
        return pipe

    chip, runtime_hw_type, compiler_options, remote_address = _prompt_compile_flags(reader, writer)
    remote = RemoteTarget(address=remote_address) if remote_address else None
    config = PipelineConfig(
        model_path=model_path,
        work_dir=work_dir,
        chip=chip,
        runtime_hw_type=runtime_hw_type,
        compiler_options=compiler_options,
    )
    if model_path.suffix in (".onnx", ".tflite"):
        writer(f"Importing {model_path} -> MLIR ...")

    pipe = ModelPipeline(config, remote)
    writer(f"Compiling for {chip} (target: {runtime_hw_type})... this can take a while for larger models.")
    try:
        compile_result = pipe.compile()
    except Exception as exc:
        pipe.write_manifest(status="error", diagnostics=str(exc))
        writer(f"Compile failed: {exc}")
        return None
    writer("Compile finished.")
    pipe.write_manifest(compile_result=compile_result)
    info = pipe._resolve_artifact_info(compile_result)
    _print_artifact_info(info, writer, header=f"Compiled: {compile_result.vmfb_path}")
    return pipe


def _initial_compile(model_path: Path, reader, writer) -> Optional[ModelPipeline]:
    work_dir = _prompt_work_dir(model_path, reader, writer)
    return _compile_with_flags(model_path, work_dir, reader, writer)


def _prompt_menu(reader, writer, commands: List[str]) -> str:
    writer("")
    writer("Menu:")
    for i, name in enumerate(commands, start=1):
        writer(f"  {i}) {name}")
    while True:
        raw = reader("Choice: ").strip().lower()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(commands):
                return commands[idx]
        elif raw in commands:
            return raw
        writer(f"Unknown choice: {raw}")


def _prompt_input_source(reader, writer):
    choice = reader("Inputs: 1) random  2) npy paths [1]: ").strip() or "1"
    if choice == "2":
        raw = reader("Input .npy paths (space-separated): ").strip()
        return False, [Path(p) for p in raw.split()]
    return True, []


def _prompt_remote(reader, writer, current: Optional[RemoteTarget]) -> Optional[RemoteTarget]:
    default = current.address if current else "local"
    raw = reader(f"Remote address [{default}]: ").strip()
    if not raw:
        return current
    return RemoteTarget(address=raw)


def _cmd_run(pipe: ModelPipeline, reader, writer) -> None:
    random_inputs, input_npy = _prompt_input_source(reader, writer)
    pipe.config.random_inputs = random_inputs
    pipe.config.input_npy = input_npy
    pipe.remote = _prompt_remote(reader, writer, pipe.remote)

    compile_result = None
    try:
        writer("Preparing the model (compiling if needed)...")
        compile_result, vmfb = pipe.ensure_vmfb()
        writer(f"Running on {_target_label(pipe)}...")
        run_result = pipe.run(vmfb)
    except Exception as exc:
        pipe.write_manifest(compile_result=compile_result, status="error", diagnostics=str(exc))
        writer(f"Run failed: {exc}")
        return
    writer("Run finished.")
    pipe.write_manifest(compile_result=compile_result, run_result=run_result, status="ok")
    writer(format_summary(_run_summary(pipe, "run", compile_result, run_result, "ok")))


def _cmd_profile(pipe: ModelPipeline, reader, writer) -> None:
    pipe.remote = _prompt_remote(reader, writer, pipe.remote)
    if not pipe.config.random_inputs and not pipe.config.input_npy:
        pipe.config.random_inputs = True
    if pipe.config.profiles_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        pipe.config.profiles_dir = Path(pipe.config.work_dir) / "profiles" / stamp
        pipe.profiles_dir = Path(pipe.config.profiles_dir).resolve()

    compile_result = None
    try:
        writer(f"Compiling if needed and profiling on {_target_label(pipe)}... this can take longer than a normal run.")
        compile_result, run_result = pipe.profile()
    except Exception as exc:
        pipe.write_manifest(compile_result=compile_result, status="error", diagnostics=str(exc))
        writer(f"Profile failed: {exc}")
        return
    writer("Profiling finished.")

    quality, detail = _profile_quality(pipe, run_result)
    if quality is None:
        diag = "profiling produced neither a host profile nor an annotated report"
        pipe.write_manifest(compile_result=compile_result, run_result=run_result, status="error", diagnostics=diag)
        writer(diag)
        return
    pipe.write_manifest(compile_result=compile_result, run_result=run_result, status="ok")
    writer(f"profile: {quality} ({detail})")
    writer(format_summary(_run_summary(pipe, "profile", compile_result, run_result, "ok")))


def _cmd_verify(pipe: ModelPipeline, reader, writer) -> None:
    raw = reader("Golden .npy paths (space-separated, blank for none): ").strip()
    goldens = [Path(p) for p in raw.split()] if raw else []
    pipe.config.expected_output_npy = goldens
    model_path = Path(pipe.config.model_path)
    if not goldens and not pipe.config.reference and model_path.suffix == ".onnx":
        pipe.config.reference = f"onnx:{model_path}"
    if not goldens and not pipe.config.reference:
        writer("no reference to verify against: provide goldens, or verify a .onnx model for a self-generated reference")
        return
    if not pipe.config.random_inputs and not pipe.config.input_npy:
        pipe.config.random_inputs = True
    pipe.remote = _prompt_remote(reader, writer, pipe.remote)

    compile_result = None
    try:
        writer("Preparing the model (compiling if needed)...")
        compile_result, vmfb = pipe.ensure_vmfb()
        writer(f"Running on {_target_label(pipe)}...")
        run_result = pipe.run(vmfb)
        writer("Comparing outputs against the reference...")
        comparison = pipe.verify_outputs(run_result)
    except Exception as exc:
        pipe.write_manifest(compile_result=compile_result, status="error", diagnostics=str(exc))
        writer(f"Verify failed: {exc}")
        return
    writer("Verification finished.")
    status = _status_for(comparison)
    pipe.write_manifest(compile_result=compile_result, run_result=run_result, comparison=comparison, status=status)
    reference = ", ".join(str(p) for p in goldens) or pipe.config.reference
    writer(format_summary(_run_summary(pipe, "verify", compile_result, run_result, status, reference=reference)))


def _cmd_inspect(pipe: ModelPipeline, reader, writer) -> None:
    info = pipe._resolve_artifact_info()
    annotated_possible = bool(
        info.debug_dir and Path(info.debug_dir).is_dir() and any(Path(info.debug_dir).iterdir())
    )
    target = info.vmfb_path or Path(pipe.config.model_path)
    hints = [
        _hint(f"run {target}"),
        _hint(f"verify {target} --golden GOLDEN.npy"),
        _hint(f"profile {target}"),
    ]
    writer(format_inspect(info, hints, annotated_possible))


def _ask_choice(reader, writer, label: str, choices: List[str], default: str) -> str:
    raw = reader(f"{label} ({'/'.join(choices)}) [{default}]: ").strip().lower()
    return raw if raw in choices else default


def _ask_yes_no(reader, writer, label: str, default: bool) -> bool:
    raw = reader(f"{label} [{'y' if default else 'n'}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _cmd_gen_config(pipe: ModelPipeline, reader, writer) -> None:
    model_path = Path(pipe.config.model_path)
    if model_path.suffix != ".onnx":
        writer("gen_config needs an .onnx model; the current model is not .onnx.")
        return
    action = _ask_choice(reader, writer, "gen_config action", _GEN_CONFIG_ACTIONS, "discover")
    argv = [action, "--model", str(model_path), "--output-dir", str(pipe.config.work_dir)]

    writer(f"Running gen_config {action}... this compiles and runs each layer, so it can take a while.")
    from torq.gen_config.cli import main as gen_config_main

    try:
        gen_config_main(argv)
    except Exception as exc:
        writer(f"gen_config {action} failed: {exc}")
        return
    writer(f"gen_config {action} finished.")


def _cmd_quantize(pipe: ModelPipeline, reader, writer) -> Optional[Path]:
    model_path = Path(pipe.config.model_path)
    if model_path.suffix != ".onnx":
        writer("quantize needs an .onnx model; the current model is not .onnx.")
        return None
    default_output = model_path.with_suffix(".int8.onnx")
    raw_output = reader(f"Output path [{default_output}]: ").strip()
    output_path = Path(raw_output) if raw_output else default_output
    num_calib = reader("--num-calib [20]: ").strip() or "20"
    per_channel = _ask_yes_no(reader, writer, "--per-channel", False)
    full_integer = _ask_yes_no(reader, writer, "--full-integer", False)

    argv = ["quantize", "--model", str(model_path), "--output", str(output_path), "--num-calib", num_calib]
    if per_channel:
        argv.append("--per-channel")
    if full_integer:
        argv.append("--full-integer")

    writer(f"Quantizing with {num_calib} calibration samples... this can take a while.")
    from torq.gen_config.cli import main as gen_config_main

    try:
        gen_config_main(argv)
    except Exception as exc:
        writer(f"Quantization failed: {exc}")
        return None
    writer(f"Quantization finished: {output_path}")

    if _ask_yes_no(reader, writer, f"Use {output_path} as the new model and recompile?", False):
        return output_path
    return None


def _menu_loop(pipe: ModelPipeline, reader, writer) -> int:
    while True:
        action = _prompt_menu(reader, writer, _MENU_ORDER)
        if action == "quit":
            return _OK
        if action == "recompile":
            model_path = Path(pipe.config.model_path)
            if model_path.suffix == ".vmfb":
                writer("No source to recompile for a .vmfb model.")
                continue
            new_pipe = _compile_with_flags(model_path, Path(pipe.config.work_dir), reader, writer)
            if new_pipe is not None:
                pipe = new_pipe
            continue
        if action == "run":
            _cmd_run(pipe, reader, writer)
        elif action == "profile":
            _cmd_profile(pipe, reader, writer)
        elif action == "verify":
            _cmd_verify(pipe, reader, writer)
        elif action == "inspect":
            _cmd_inspect(pipe, reader, writer)
        elif action == "gen_config":
            _cmd_gen_config(pipe, reader, writer)
        elif action == "quantize":
            new_model = _cmd_quantize(pipe, reader, writer)
            if new_model is not None:
                new_pipe = _compile_with_flags(new_model, Path(pipe.config.work_dir), reader, writer)
                if new_pipe is not None:
                    pipe = new_pipe
