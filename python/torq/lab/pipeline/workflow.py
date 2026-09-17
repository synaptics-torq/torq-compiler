# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ModelPipeline: compile/run/profile orchestration over local and remote targets.

This module owns the pipeline configuration and result types
(``PipelineConfig``, ``CompileResult``, ``RunResult``) alongside the
orchestration that produces them.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from torq.lab import LabError
from torq.lab.pipeline import artifacts, io, tools
from torq.lab.pipeline.io import ConvertIODTypesPolicy
from torq.lab.pipeline.remote import RemoteTarget

logger = logging.getLogger("torq.lab.pipeline")


def _s(p):
    """Path -> str, preserving None (for JSON serialization)."""
    return None if p is None else str(p)


def _p(v):
    """str -> Path, preserving None (for JSON deserialization)."""
    return Path(v) if v is not None else None


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base`` (overlay wins); returns a new dict."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(sources) -> dict:
    """Deep-merge an ordered list of JSON config files (or dicts); later wins.

    Each source may be a bare config dict or a full manifest (its ``config``
    section is used). This enables docker-compose-style layering: a global base
    (model, chip, compiler options) plus a local overlay (e.g. the board's
    remote target). Feed the result to :meth:`PipelineConfig.from_dict`.
    """
    merged: dict = {}
    for source in sources:
        if isinstance(source, dict):
            data = source
        else:
            path = Path(source)
            try:
                text = path.read_text()
            except OSError as exc:
                raise LabError(f"could not read config file {path}: {exc}") from exc
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LabError(f"invalid JSON in config file {path}: {exc}") from exc
        if isinstance(data, dict):
            data = data.get("config", data)
        merged = _deep_merge(merged, data)
    return merged

@dataclass
class PipelineConfig:
    """Everything needed to compile and/or run a model.

    ``model_path`` may be ``.onnx`` or ``.tflite`` (imported to MLIR at compile
    time via ``ModelPipeline.ensure_mlir()``), ``.mlir``, or ``.vmfb`` (run
    only). For compile/compile-run it can be an ``.onnx``, ``.tflite``, or a
    ``.mlir``. For run, the VMFB is resolved
    by precedence: explicit run() argument, config.vmfb_path, a .vmfb model_path,
    then <work-dir>/model.vmfb. The IO spec is derived from ``model_path`` itself
    (an ``.mlir``) or its sibling ``.mlir`` next to a ``.vmfb``; ``spec_source``
    is a library-level override for the rare case the spec lives elsewhere. It
    has no CLI flag (the CLI relies on the model-path convention).

    ``input_seed`` and ``input_ranges`` steer the ``random_inputs`` generator
    (:func:`torq.lab.pipeline.io.generate_random_inputs`): ``input_seed`` selects the RNG
    stream (``None`` keeps the historical default of 1234), and ``input_ranges``
    maps an input name (when the spec carries one) or its decimal index
    (``"0"``, ``"1"``, ...) to a ``(min, max)`` range that input's values are
    drawn from. Both are ``None`` by default, which reproduces the historical
    full-range behavior byte-for-byte.

    ``input_specs`` and ``output_specs`` are explicit tensor-type strings
    (e.g., ``["1x1x64xbf16"]``) that supply the I/O signature when neither MLIR
    nor a manifest is available. They feed the artifact resolver's explicit-specs
    tier (precedence 5), allowing random input generation and typed output
    capture for delivered VMFBs with no source.
    """

    model_path: Path
    work_dir: Path
    chip: str = "SL2610"
    runtime_hw_type: str = "sim"
    function: Optional[str] = None
    compiler_options: List[str] = field(default_factory=list)
    runtime_options: List[str] = field(default_factory=list)
    dump_ir: bool = False
    dump_phases: bool = False
    profile_compile: bool = False
    profile_runtime: bool = False
    convert_io_dtypes: List[str] = field(default_factory=list)
    random_inputs: bool = False
    input_npy: List[Path] = field(default_factory=list)
    expected_output_npy: List[Path] = field(default_factory=list)
    reference: Optional[str] = None
    timeout: Optional[int] = None
    spec_source: Optional[Path] = None
    vmfb_path: Optional[Path] = None
    compile_tool: Optional[str] = None
    run_tool: Optional[str] = None
    input_seed: Optional[int] = None
    input_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    profiles_dir: Optional[Path] = None
    input_specs: List[str] = field(default_factory=list)
    output_specs: List[str] = field(default_factory=list)
    reuse_work_dir: bool = False

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (Paths -> str)."""
        return {
            "model_path": _s(self.model_path),
            "work_dir": _s(self.work_dir),
            "chip": self.chip,
            "runtime_hw_type": self.runtime_hw_type,
            "function": self.function,
            "compiler_options": list(self.compiler_options),
            "runtime_options": list(self.runtime_options),
            "dump_ir": self.dump_ir,
            "dump_phases": self.dump_phases,
            "profile_compile": self.profile_compile,
            "profile_runtime": self.profile_runtime,
            "convert_io_dtypes": list(self.convert_io_dtypes),
            "random_inputs": self.random_inputs,
            "input_npy": [_s(p) for p in self.input_npy],
            "expected_output_npy": [_s(p) for p in self.expected_output_npy],
            "reference": self.reference,
            "timeout": self.timeout,
            "spec_source": _s(self.spec_source),
            "vmfb_path": _s(self.vmfb_path),
            "compile_tool": self.compile_tool,
            "run_tool": self.run_tool,
            "input_seed": self.input_seed,
            "input_ranges": (
                None
                if self.input_ranges is None
                else {k: [v[0], v[1]] for k, v in self.input_ranges.items()}
            ),
            "profiles_dir": _s(self.profiles_dir),
            "input_specs": list(self.input_specs),
            "output_specs": list(self.output_specs),
            "reuse_work_dir": self.reuse_work_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Reconstruct from a config dict (as produced by :meth:`to_dict`).

        ``model_path`` is required; ``work_dir`` defaults to
        ``./<model-stem>-run`` when absent, matching the CLI convention.
        Unknown keys (e.g. a nested ``remote`` section) are ignored.
        """
        if not d.get("model_path"):
            raise LabError("config is missing required 'model_path'")
        model_path = _p(d["model_path"])
        work_dir = _p(d.get("work_dir")) or (Path.cwd() / f"{model_path.stem}-run")
        return cls(
            model_path=model_path,
            work_dir=work_dir,
            chip=d.get("chip", "SL2610"),
            runtime_hw_type=d.get("runtime_hw_type", "sim"),
            function=d.get("function"),
            compiler_options=list(d.get("compiler_options", [])),
            runtime_options=list(d.get("runtime_options", [])),
            dump_ir=d.get("dump_ir", False),
            dump_phases=d.get("dump_phases", False),
            profile_compile=d.get("profile_compile", False),
            profile_runtime=d.get("profile_runtime", False),
            convert_io_dtypes=list(d.get("convert_io_dtypes", [])),
            random_inputs=d.get("random_inputs", False),
            input_npy=[_p(p) for p in d.get("input_npy", [])],
            expected_output_npy=[_p(p) for p in d.get("expected_output_npy", [])],
            reference=d.get("reference"),
            timeout=d.get("timeout"),
            spec_source=_p(d.get("spec_source")),
            vmfb_path=_p(d.get("vmfb_path")),
            compile_tool=d.get("compile_tool"),
            run_tool=d.get("run_tool"),
            input_seed=d.get("input_seed"),
            input_ranges=(
                None
                if d.get("input_ranges") is None
                else {str(k): tuple(v) for k, v in d["input_ranges"].items()}
            ),
            profiles_dir=_p(d.get("profiles_dir")),
            input_specs=list(d.get("input_specs", [])),
            output_specs=list(d.get("output_specs", [])),
            reuse_work_dir=d.get("reuse_work_dir", False),
        )

    @classmethod
    def from_manifest(cls, manifest: dict) -> "PipelineConfig":
        """Reconstruct from a manifest (v2 ``config`` section; v1 flat layout)."""
        return cls.from_dict(manifest.get("config", manifest))

    @classmethod
    def from_file(cls, path) -> "PipelineConfig":
        """Reconstruct from a manifest/config JSON file."""
        return cls.from_manifest(json.loads(Path(path).read_text()))

@dataclass
class CompileResult:
    """What :meth:`ModelPipeline.compile <torq.lab.pipeline.workflow.ModelPipeline.compile>` produced.

    ``vmfb_path`` is the compiled module, ``command`` the exact ``torq-compile``
    argv, and ``elapsed`` the wall-clock compile time in seconds. The optional
    paths point at artifacts emitted only when the matching config flag is set:
    ``debug_dir`` (always, for profiling annotation), ``phases_dir``
    (``dump_phases``) and, under ``profile_compile``, ``compile_profile`` (the
    raw CSV), ``compile_trace`` (the Perfetto ``.pb``) and ``perfetto_viewer``
    (the self-contained HTML viewer). ``diagnostics`` carries the tool's captured
    stderr.
    """

    vmfb_path: Path
    command: List[str]
    elapsed: float
    debug_dir: Optional[Path] = None
    phases_dir: Optional[Path] = None
    compile_profile: Optional[Path] = None
    compile_trace: Optional[Path] = None
    perfetto_viewer: Optional[Path] = None
    diagnostics: str = ""


@dataclass
class RunResult:
    """What :meth:`ModelPipeline.run <torq.lab.pipeline.workflow.ModelPipeline.run>` produced.

    ``command`` is the exact ``torq-run-module`` argv (or the remote command).
    ``output_paths`` are the raw ``.bin`` files written by the run and
    ``outputs`` the same data loaded into numpy arrays (empty when the IO spec is
    unknown). ``wall_time`` is the end-to-end time in seconds. The profiling
    fields (``host_profile``, ``annotated_profile``, ``trace``,
    ``perfetto_viewer``) are populated only under a profiling mode; see
    :mod:`torq.lab.profiling`. ``diagnostics`` carries the run's output/stderr.
    """

    command: List[str]
    output_paths: List[Path] = field(default_factory=list)
    outputs: List[Any] = field(default_factory=list)
    host_profile: Optional[Path] = None
    annotated_profile: Optional[Path] = None
    trace: Optional[Path] = None
    perfetto_viewer: Optional[Path] = None
    wall_time: Optional[float] = None
    diagnostics: str = ""


def build_compile_command(config, tool, source, vmfb_path, debug_dir, phases_dir, compile_profile) -> List[str]:
    """Assemble the ``torq-compile`` command line from a PipelineConfig."""
    cmds = [str(tool), str(Path(source).resolve()), "-o", str(vmfb_path)]
    cmds.append(f"--torq-hw={config.chip}")

    if config.dump_ir:
        ir_dir = Path(debug_dir) / "ir"
        cmds.extend([
            "--mlir-print-ir-after-all",
            f"--mlir-print-ir-tree-dir={ir_dir}",
            "--mlir-elide-elementsattrs-if-larger=50",
        ])

    if config.dump_phases:
        cmds.append(f"--dump-compilation-phases-to={phases_dir}")

    # The cmodel/fpga host needs native host binaries
    if config.runtime_hw_type in ["sim", "aws_fpga"]:
        cmds.extend(["--torq-target-host-triple=native"])    

    cmds += list(config.compiler_options)

    # Debug info is always emitted; profiling annotates cycle-time attributes.
    cmds.append(f"--torq-debug-info={debug_dir}")
    if config.profile_compile:
        cmds.extend(["--torq-enable-profiling", f"--torq-dump-profiling={compile_profile}"])
    elif config.profile_runtime:
        cmds.append("--torq-enable-profiling")

    return cmds


def build_run_command(config, tool, vmfb_path, func_name, output_args, input_args, host_profile) -> List[str]:
    """Assemble the ``torq-run-module`` command line from a PipelineConfig."""
    cmds = [
        str(tool),
        f"--module={vmfb_path}",
        f"--function={func_name}",
        *output_args,
        f"--torq_hw_type={config.runtime_hw_type}",
        *list(config.runtime_options),
        *input_args,
    ]
    if config.profile_runtime:
        cmds.append(f"--torq_profile_host={host_profile}")
    return cmds


def _diagnostics(proc) -> str:
    try:
        return proc.stderr.decode("utf-8", errors="replace")
    except Exception:
        return ""


class ModelPipeline:
    """High-level compile/run/profile orchestration around a PipelineConfig."""

    def __init__(self, config: PipelineConfig, remote: Optional[RemoteTarget] = None):
        self.config = config
        self.remote = remote
        # Tools run with this directory as their cwd, so artifact paths must
        # remain anchored to the caller's cwd rather than becoming nested.
        self.work_dir = Path(config.work_dir).resolve()
        self.inputs_dir = self.work_dir / "inputs"
        self.outputs_dir = self.work_dir / "outputs"
        self.debug_dir = self.work_dir / "debug"
        self.phases_dir = self.work_dir / "phases"
        self.profiles_dir = Path(config.profiles_dir).resolve() if config.profiles_dir else self.work_dir / "profiles"
        self.remote_dir = self.work_dir / "remote"

    # -- helpers ---------------------------------------------------------

    def _ensure_work_dir(self) -> None:
        """Create the work directory, allocating a run ID if it has run artifacts."""
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if self.config.reuse_work_dir:
            return

        has_run_artifacts = any([
            (self.work_dir / "inputs").exists(),
            (self.work_dir / "outputs").exists(),
        ])
        if not has_run_artifacts:
            return

        import uuid
        run_id = uuid.uuid4().hex[:8]
        original_work_dir = self.work_dir
        self.work_dir = original_work_dir.parent / f"{original_work_dir.name}_{run_id}"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.inputs_dir = self.work_dir / "inputs"
        self.outputs_dir = self.work_dir / "outputs"
        self.debug_dir = self.work_dir / "debug"
        self.phases_dir = self.work_dir / "phases"
        self.remote_dir = self.work_dir / "remote"
        # Preserve a caller-configured (e.g. timestamped) profiles dir across
        # reallocation; only the work-dir default tracks the new work dir.
        self.profiles_dir = (
            Path(self.config.profiles_dir).resolve()
            if self.config.profiles_dir
            else self.work_dir / "profiles"
        )

    def _vmfb_path(self) -> Path:
        """Resolve the VMFB path by precedence: explicit run() argument, config.vmfb_path, a .vmfb model_path, then <model-stem>.vmfb in the work dir."""
        if self.config.vmfb_path:
            return Path(self.config.vmfb_path).resolve()
        if Path(self.config.model_path).suffix == ".vmfb":
            return Path(self.config.model_path).resolve()
        return self.work_dir / f"{Path(self.config.model_path).stem}.vmfb"

    def _explicit_source(self) -> Optional[Path]:
        """The caller-known source hint for ``artifact.describe()``.

        ``config.spec_source`` (set for imported ONNX/TFLite), else
        ``model_path`` itself when it is already an ``.mlir``, else ``None``.
        """
        if self.config.spec_source:
            return Path(self.config.spec_source)
        model_path = Path(self.config.model_path)
        return model_path if model_path.suffix == ".mlir" else None

    def _resolve_artifact_info(self, compile_result: Optional[CompileResult] = None) -> artifacts.ArtifactInfo:
        """Resolve this pipeline's own artifact facts for the manifest's ``artifact`` section."""
        info = artifacts.describe(self._vmfb_path(), source=self._explicit_source())
        info.chip = info.chip or self.config.chip
        if compile_result is not None:
            info.debug_dir = info.debug_dir or compile_result.debug_dir
            info.compile_command = info.compile_command or list(compile_result.command)
        if self._convert_io_dtypes_policy() and not info.converted_io:
            info.converted_io = list(self.config.convert_io_dtypes) or ["all"]
        return info

    def _convert_io_dtypes_policy(self) -> ConvertIODTypesPolicy:
        if self.config.convert_io_dtypes:
            return ConvertIODTypesPolicy.parse_from_args(self.config.convert_io_dtypes, True)
        enabled = {"--torq-convert-dtypes", "--torq-convert-io-dtype"}.issubset(
            self.config.compiler_options
        )
        return ConvertIODTypesPolicy.parse_from_args(["all"], enabled)

    def _materialize_inputs(
        self, spec, convert_io_dtypes_policy: ConvertIODTypesPolicy, info: artifacts.ArtifactInfo = None
    ) -> Tuple[List[np.ndarray], List[Path]]:
        if self.config.input_npy:
            inputs = [io.load_npy(p) for p in self.config.input_npy]
        elif self.config.random_inputs:
            if spec is None:
                recovery = self._input_recovery_options(info)
                raise LabError(f"Cannot generate inputs for {Path(self.config.model_path).name}: no input signature is available.\n\nProvide one of:\n{recovery}")
            inputs = io.generate_random_inputs(
                spec, seed=self.config.input_seed, ranges=self.config.input_ranges
            )
        else:
            return [], []
        # Match the dtype the runtime will actually see: each input is cast to
        # its spec dtype (after the convert-io policy), so e.g. an f32 .npy
        # for a bf16-IO model is not written verbatim (QA M2).
        if spec is not None:
            runtime_spec = convert_io_dtypes_policy.convert_io_spec(spec)
            inputs = [
                data.astype(io.get_dtype(runtime_spec.inputs[idx].fmt))
                if idx < len(runtime_spec.inputs)
                and np.dtype(data.dtype) != io.get_dtype(runtime_spec.inputs[idx].fmt)
                else data
                for idx, data in enumerate(inputs)
            ]
        paths = io.write_inputs(inputs, self.inputs_dir)
        return inputs, paths

    def _input_recovery_options(self, info: artifacts.ArtifactInfo) -> str:
        """Format recovery command hints from the artifact's provenance."""
        from torq.lab.cli.parser import PROG
        model_path = Path(self.config.model_path)
        hints = []
        vmfb = model_path if model_path.suffix == ".vmfb" else self._vmfb_path()
        hints.append(f"  {PROG} run {vmfb} --input-npy input_0.npy")
        hints.append(f"  {PROG} run {vmfb} --input-spec '1x1x64xbf16' --random-inputs")
        if info and info.source_path:
            hints.append(f"  {PROG} run {vmfb} --source {info.source_path}")
        return "\n".join(hints)

    def _collect_profiles(self):
        host = self.profiles_dir / "host_profile.csv"
        host = host if host.exists() else None
        annotated = trace = None
        if self.profiles_dir.exists():
            xlsx = sorted(self.profiles_dir.glob("annotated_profile*.xlsx"))
            annotated = xlsx[0] if xlsx else None
            pbs = sorted(self.profiles_dir.glob("trace*.pb"))
            trace = pbs[0] if pbs else None
        return host, annotated, trace

    def _collect_compile_profiles(self):
        """Locate the compile-time Perfetto trace and viewer, if produced."""
        trace = None
        if self.profiles_dir.exists():
            pbs = sorted(self.profiles_dir.glob("*_compile.pb"))
            trace = pbs[0] if pbs else None
        return trace, self._viewer_path()

    def _viewer_path(self) -> Optional[Path]:
        viewer = self.profiles_dir / "perfetto_viewer.html"
        return viewer if viewer.exists() else None

    def _has_debug_info(self) -> bool:
        """True when the compile emitted debug info to annotate against."""
        return self.debug_dir.is_dir() and any(self.debug_dir.iterdir())

    def _finalize_profiles(self) -> None:
        """Annotate the host profile, add compile trace(s), and render the viewer.

        No-op unless a profiling mode is enabled and the compile produced debug
        info to annotate against. Requires the ``[profile]`` extra; without it
        the profile stays raw (host CSV only) and a warning is logged rather
        than failing the run — see ``_profile_quality`` in the lab CLI.

        Safe to call after both ``compile`` (compile trace) and ``run`` (host
        annotation): the compile trace is written only once, so a subsequent run
        adds the annotated run trace and re-renders a combined viewer.
        """
        if not (self.config.profile_runtime or self.config.profile_compile):
            return
        if not self._has_debug_info():
            return
        from torq.lab.profiling import annotate

        try:
            host_profile = self.profiles_dir / "host_profile.csv"
            if self.config.profile_runtime and host_profile.exists():
                annotate.annotate_run_profile(self.debug_dir, host_profile, self.profiles_dir)
            if self.config.profile_compile and not any(self.profiles_dir.glob("*_compile.pb")):
                annotate.write_compile_trace(self.debug_dir, self.profiles_dir)

            pb_files = sorted(self.profiles_dir.glob("*.pb"))
            if pb_files:
                annotate.write_perfetto_report(pb_files, self.profiles_dir / "perfetto_viewer.html")
        except LabError as exc:
            # Without the [profile] extra the helpers raise LabError before
            # writing anything; keep the raw host profile and warn rather than
            # failing the run.
            logger.warning("%s; keeping the raw host profile", exc)

    # -- stages ----------------------------------------------------------

    def ensure_mlir(self) -> Path:
        """Return the ``.mlir`` source to compile, importing ONNX/TFLite if needed.

        ``.mlir`` -> returned unchanged. ``.onnx`` -> imported via
        ``torq.lab.model_tools.importers.onnx.convert_onnx_to_mlir``. ``.tflite`` -> imported via
        ``torq.lab.model_tools.importers.tflite.convert_tflite_to_mlir``. Both write
        ``<work-dir>/<model-stem>.mlir`` and return that path; the imported path
        is recorded as ``spec_source`` so a later run resolves I/O from it.
        ``.vmfb`` (or any other suffix) -> raises ``LabError`` naming the
        accepted suffixes: compile needs a source, not a module.
        """
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise LabError(f"no such model file: {model_path}")
        if model_path.suffix == ".mlir":
            return model_path.resolve()
        if model_path.suffix == ".onnx":
            from torq.lab.model_tools.importers.onnx import convert_onnx_to_mlir

            return self._import_to_mlir(model_path, convert_onnx_to_mlir, "ONNX")
        if model_path.suffix == ".tflite":
            from torq.lab.model_tools.importers.tflite import convert_tflite_to_mlir

            return self._import_to_mlir(model_path, convert_tflite_to_mlir, "TFLite")
        raise LabError(f"compile needs an .onnx, .tflite, or .mlir source, got {model_path}")

    def _import_to_mlir(self, model_path: Path, importer, label: str) -> Path:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        mlir = self.work_dir / f"{model_path.stem}.mlir"
        try:
            importer(model_path.resolve(), mlir, timeout=self.config.timeout or 300)
        except RuntimeError as exc:
            raise LabError(f"failed to import {label} {model_path}: {exc}")
        if self.config.spec_source is None:
            self.config.spec_source = mlir
        return mlir

    def _dynamic_shape_error(self, detail: str) -> LabError:
        """Build the user-facing error for a dynamic-shape model.

        ``detail`` names what was found, e.g.
        ``"dyn.tflite has dynamic dimension(s) in its entry function I/O: ..."``.
        """
        model = Path(self.config.model_path)
        message = f"the TORQ compiler does not support dynamic shapes: {detail}"
        if model.suffix == ".tflite":
            static = model.with_name(f"{model.stem}_static.tflite")
            message += (
                f"\n\nConvert the model to static shapes first (requires the [tf] extra) "
                f"and run that instead:\n"
                f"  torq-convert-static tflite -i {model} -o {static}"
            )
        else:
            message += "\n\nRe-export the model with static shapes and try again."
        return LabError(message)

    def compile(self) -> CompileResult:
        """Compile ``config.model_path`` (an ``.mlir``) to a VMFB.

        Creates the work-dir layout, invokes ``torq-compile`` with flags derived
        from the config (chip, IR/phase dumps, profiling), and returns a
        :class:`~torq.lab.pipeline.workflow.CompileResult`. Raises
        :class:`~torq.lab.LabError` / ``ToolError`` on tool failure.
        """
        source = self.ensure_mlir()
        # The TORQ backend requires fully static shapes; fail before invoking
        # torq-compile with an actionable message instead of a raw compiler
        # error (or an MLIR assertion abort).
        dynamic = io.dynamic_io_types(source, self.config.function)
        if dynamic:
            types = ", ".join(sorted(set(dynamic)))
            raise self._dynamic_shape_error(
                f"{Path(self.config.model_path).name} has dynamic dimension(s) in its "
                f"entry function I/O: {types}"
            )
        self._ensure_work_dir()
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        if self.config.dump_ir:
            (self.debug_dir / "ir").mkdir(parents=True, exist_ok=True)
        if self.config.dump_phases:
            self.phases_dir.mkdir(parents=True, exist_ok=True)
        compile_profile = None
        if self.config.profile_compile:
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            compile_profile = self.profiles_dir / "compile_profile.csv"

        tool = tools.find_compile_tool(self.config.compile_tool)
        vmfb = self._vmfb_path()
        cmds = build_compile_command(
            self.config, tool, source, vmfb, self.debug_dir, self.phases_dir, compile_profile
        )
        start = time.perf_counter()
        try:
            proc = tools.run_tool(cmds, timeout=self.config.timeout, cwd=self.work_dir)
        except tools.ToolError as exc:
            # Internal (non-I/O) dynamic dimensions reach the backend as a
            # generic error; map it to the same actionable message.
            if "unhandled dynamic dimensions" in exc.output:
                raise self._dynamic_shape_error(
                    f"dynamic dimensions were detected in {Path(self.config.model_path).name} "
                    f"during compilation"
                ) from exc
            raise
        elapsed = time.perf_counter() - start

        # Compile-time profiling produces its trace and viewer here, without
        # needing a run (unlike runtime profiling, which is finalized in run()).
        compile_trace = perfetto_viewer = None
        if self.config.profile_compile:
            self._finalize_profiles()
            compile_trace, perfetto_viewer = self._collect_compile_profiles()

        return CompileResult(
            vmfb_path=vmfb,
            command=cmds,
            elapsed=elapsed,
            debug_dir=self.debug_dir,
            phases_dir=self.phases_dir if self.config.dump_phases else None,
            compile_profile=compile_profile,
            compile_trace=compile_trace,
            perfetto_viewer=perfetto_viewer,
            diagnostics=_diagnostics(proc),
        )

    def ensure_vmfb(self) -> Tuple[Optional[CompileResult], Path]:
        """Return the VMFB to execute, compiling a source first if needed.

        ``.onnx``/``.tflite``/``.mlir`` -> ``self.compile()`` (which imports
        ONNX/TFLite via ``ensure_mlir``), returning ``(compile_result,
        compile_result.vmfb_path)``. ``.vmfb`` -> ``(None, self._vmfb_path())``.
        Any other suffix -> ``LabError`` naming the accepted suffixes and the
        path given.
        """
        model_path = Path(self.config.model_path)
        if model_path.suffix == ".vmfb":
            if not model_path.exists():
                raise LabError(f"no such model file: {model_path}")
            return None, self._vmfb_path()
        if model_path.suffix in (".onnx", ".tflite", ".mlir"):
            compile_result = self.compile()
            return compile_result, compile_result.vmfb_path
        raise LabError(
            f"run/verify/profile need an .onnx, .tflite, .mlir, or .vmfb model, got {model_path}"
        )

    def run(self, vmfb_path=None) -> RunResult:
        """Run a compiled VMFB and collect its outputs.

        Uses ``vmfb_path`` when given, else ``config.vmfb_path`` or
        ``<work-dir>/model.vmfb``. Materializes inputs (from ``input_npy`` or
        ``random_inputs``), resolves the entry function and output specs from the
        MLIR IO spec, executes locally or on ``self.remote``, loads outputs back
        into numpy arrays, and finalizes any profiling artifacts. Returns a
        :class:`~torq.lab.pipeline.workflow.RunResult`.
        """
        self._ensure_work_dir()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        vmfb = Path(vmfb_path) if vmfb_path else self._vmfb_path()

        # Validate user-supplied spec strings up front: when a manifest or
        # sibling source already resolves the I/O spec, the explicit specs are
        # not parsed (artifacts.describe's last-resort tier), so a typo'd
        # dtype would otherwise be silently ignored.
        for spec_str in (*self.config.input_specs, *self.config.output_specs):
            io.TensorType.from_string(spec_str)

        info = artifacts.describe(
            vmfb, source=self._explicit_source(),
            input_specs=self.config.input_specs, output_specs=self.config.output_specs
        )
        spec = info.io_spec
        func_name = self.config.function or info.function

        convert_io_dtypes_policy = self._convert_io_dtypes_policy()
        runtime_spec = convert_io_dtypes_policy.convert_io_spec(spec) if spec else None
        inputs, input_paths = self._materialize_inputs(spec, convert_io_dtypes_policy, info)
        input_args = io.build_input_args(input_paths, inputs, runtime_spec) if inputs else []

        if runtime_spec is not None:
            output_args = io.create_output_args(self.outputs_dir, runtime_spec.outputs)
            output_paths = [Path(p) for p in io.create_output_paths(self.outputs_dir, runtime_spec.outputs)]
        else:
            output_args, output_paths = [], []

        host_profile = None
        if self.config.profile_runtime:
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            host_profile = self.profiles_dir / "host_profile.csv"

        if self.remote is not None:
            return self._run_remote(vmfb, func_name, input_args, input_paths, output_args, output_paths, runtime_spec)

        tool = tools.find_run_tool(self.config.run_tool)
        cmds = build_run_command(
            self.config, tool, vmfb, func_name, output_args, input_args, host_profile
        )
        start = time.perf_counter()
        proc = tools.run_tool(cmds, timeout=self.config.timeout, cwd=self.work_dir)
        wall = time.perf_counter() - start

        outputs = io.load_outputs(runtime_spec.outputs, output_paths) if (runtime_spec and output_paths) else []
        self._finalize_profiles()
        host, annotated, trace = self._collect_profiles()
        return RunResult(
            command=cmds,
            output_paths=output_paths,
            outputs=outputs,
            host_profile=host,
            annotated_profile=annotated,
            trace=trace,
            perfetto_viewer=self._viewer_path(),
            wall_time=wall,
            diagnostics=_diagnostics(proc),
        )

    def _run_remote(self, vmfb, func_name, input_args, input_paths, output_args, output_paths, spec) -> RunResult:
        from torq.lab.pipeline.remote import RemoteExecutor

        self.remote_dir.mkdir(parents=True, exist_ok=True)
        runtime_opts = [f"--torq_hw_type={self.config.runtime_hw_type}", *list(self.config.runtime_options)]
        if self.config.profile_runtime:
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            runtime_opts.append(f"--torq_profile_host={self.profiles_dir / 'host_profile.csv'}")

        executor = RemoteExecutor(
            self.remote, vmfb, func_name, input_args, output_args, runtime_opts,
            timeout=self.config.timeout,
        )
        outcome = executor.run()

        outputs = io.load_outputs(spec.outputs, output_paths) if (spec and output_paths) else []
        self._finalize_profiles()
        host, annotated, trace = self._collect_profiles()
        return RunResult(
            command=outcome.command,
            output_paths=output_paths,
            outputs=outputs,
            host_profile=host,
            annotated_profile=annotated,
            trace=trace,
            perfetto_viewer=self._viewer_path(),
            wall_time=outcome.wall_time,
            diagnostics=outcome.output,
        )

    def compile_run(self) -> Tuple[CompileResult, RunResult]:
        """Compile the model, then run the freshly built VMFB in one call."""
        compile_result = self.compile()
        run_result = self.run(compile_result.vmfb_path)
        return compile_result, run_result

    def profile(self) -> Tuple[Optional[CompileResult], RunResult]:
        """Compile if needed, run with host profiling on, and finalize the profile.

        Sets ``profile_runtime`` before ``ensure_vmfb()``: a source that needs
        compiling must compile with ``--torq-enable-profiling`` already on
        (``build_compile_command`` gates that flag on ``profile_runtime``), so
        profiling a source has to compile with it on, not just run with it.
        """
        self.config.profile_runtime = True
        compile_result, vmfb = self.ensure_vmfb()
        run_result = self.run(vmfb)
        return compile_result, run_result

    # -- reporting -------------------------------------------------------

    def expected_outputs(self) -> Optional[List[np.ndarray]]:
        """Load the ``expected_output_npy`` arrays, or ``None`` if none configured."""
        if not self.config.expected_output_npy:
            return None
        return [io.load_npy(p) for p in self.config.expected_output_npy]

    def compare(self, run_result: RunResult):
        """Compare run outputs against the configured expected outputs, if any.

        Returns ``None`` only when no expected outputs are configured. When they
        are, the comparison always runs, so a run that produced no outputs (or a
        mismatched output count) is reported as a failure instead of silently
        passing.
        """
        expected = self.expected_outputs()
        if expected is None:
            return None
        from torq.lab.verification.compare import compare_outputs

        return compare_outputs(run_result.outputs, expected)

    def _loaded_run_inputs(self) -> List[np.ndarray]:
        """Load the input arrays the most recent run used, from ``self.inputs_dir``, in index order."""
        paths = sorted(
            self.inputs_dir.glob("in_rnd_*.bin.npy"),
            key=lambda p: int(p.name[len("in_rnd_"):-len(".bin.npy")]),
        )
        return [io.load_npy(p) for p in paths]

    def reference_outputs(self) -> Optional[List[np.ndarray]]:
        """Generate expected outputs from ``config.reference``, or None if unset.

        Loads the inputs the run used (from ``self.inputs_dir``) and calls
        ``reference.onnx_reference_outputs`` on the resolved ONNX path. Uses the
        same input files the VMFB saw so the only difference is the compute path.
        """
        if not self.config.reference:
            return None
        provider, _, path = self.config.reference.partition(":")
        if provider != "onnx":
            raise LabError(f"unknown --reference provider '{provider}'; only 'onnx' is supported")
        if path:
            onnx_path = Path(path)
        else:
            model_path = Path(self.config.model_path)
            if model_path.suffix != ".onnx":
                raise LabError("--reference onnx needs an .onnx model; give --reference onnx:PATH")
            onnx_path = model_path

        from torq.lab.verification.reference import onnx_reference_outputs

        return onnx_reference_outputs(onnx_path, self._loaded_run_inputs())

    def verify_outputs(self, run_result: RunResult):
        """Compare run outputs against goldens if configured, else a generated
        reference, else None. Prefers explicit goldens over generation."""
        expected = self.expected_outputs()
        if expected is None:
            expected = self.reference_outputs()
        if expected is None:
            return None
        from torq.lab.verification.compare import compare_outputs

        return compare_outputs(run_result.outputs, expected)

    def write_manifest(self, **kwargs) -> Path:
        """Build and write ``manifest.json`` into the work dir, including this pipeline's artifact facts."""
        from torq.lab.pipeline.artifacts import build_manifest
        from torq.lab.pipeline.artifacts import write_manifest as _write

        info = self._resolve_artifact_info(kwargs.get("compile_result"))
        manifest = build_manifest(config=self.config, remote=self.remote, artifact=info, **kwargs)
        return _write(self.work_dir, manifest)
