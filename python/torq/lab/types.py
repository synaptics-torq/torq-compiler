# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public dataclasses and the base exception for torq.lab."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


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
        data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
        if isinstance(data, dict):
            data = data.get("config", data)
        merged = _deep_merge(merged, data)
    return merged


class LabError(Exception):
    """Base class for torq.lab errors."""


@dataclass
class TensorType:
    """The type of a tensor input or output of an MLIR model."""

    shape: List[int]
    fmt: str

    def to_arg(self) -> str:
        return "x".join([str(x) for x in self.shape] + [self.fmt])

    @staticmethod
    def from_string(spec: str) -> "TensorType":
        *shape_str, fmt = spec.split("x")
        shape = [int(s) for s in shape_str]
        return TensorType(shape, fmt)


@dataclass
class MlirIoSpec:
    """Input and output tensor types parsed from an MLIR module."""

    inputs: List[TensorType]
    outputs: List[TensorType]


@dataclass
class PipelineConfig:
    """Everything needed to compile and/or run a model.

    ``model_path`` is the ``.mlir`` source for compile/compile-run and the
    ``.vmfb`` module for run. The IO spec is derived from ``model_path`` itself
    (an ``.mlir``) or its sibling ``.mlir`` next to a ``.vmfb``; ``spec_source``
    is a library-level override for the rare case the spec lives elsewhere. It
    has no CLI flag (the CLI relies on the model-path convention).
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
    random_inputs: bool = False
    input_npy: List[Path] = field(default_factory=list)
    expected_output_npy: List[Path] = field(default_factory=list)
    timeout: Optional[int] = None
    spec_source: Optional[Path] = None
    vmfb_path: Optional[Path] = None
    compile_tool: Optional[str] = None
    run_tool: Optional[str] = None

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
            "random_inputs": self.random_inputs,
            "input_npy": [_s(p) for p in self.input_npy],
            "expected_output_npy": [_s(p) for p in self.expected_output_npy],
            "timeout": self.timeout,
            "spec_source": _s(self.spec_source),
            "vmfb_path": _s(self.vmfb_path),
            "compile_tool": self.compile_tool,
            "run_tool": self.run_tool,
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
            random_inputs=d.get("random_inputs", False),
            input_npy=[_p(p) for p in d.get("input_npy", [])],
            expected_output_npy=[_p(p) for p in d.get("expected_output_npy", [])],
            timeout=d.get("timeout"),
            spec_source=_p(d.get("spec_source")),
            vmfb_path=_p(d.get("vmfb_path")),
            compile_tool=d.get("compile_tool"),
            run_tool=d.get("run_tool"),
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
class RemoteTarget:
    """A remote board reachable over SSH or ADB.

    ``address`` accepts the same forms as the transport factory: ``adb`` (first
    device), an ADB serial, ``user@host``, or a bare hostname/IP for SSH.
    """

    address: str
    port: int = 22
    private_key: Optional[str] = None
    remote_runner_path: Optional[str] = None
    stage_runner: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "address": self.address,
            "port": self.port,
            "private_key": self.private_key,
            "remote_runner_path": self.remote_runner_path,
            "stage_runner": self.stage_runner,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RemoteTarget":
        return cls(
            address=d["address"],
            port=d.get("port", 22),
            private_key=d.get("private_key"),
            remote_runner_path=d.get("remote_runner_path"),
            stage_runner=d.get("stage_runner"),
        )

    @classmethod
    def from_manifest(cls, manifest: dict) -> Optional["RemoteTarget"]:
        """Reconstruct the remote target from a manifest's config section, if any."""
        section = manifest.get("config", manifest)
        remote = section.get("remote")
        return cls.from_dict(remote) if remote else None


@dataclass
class CompileResult:
    """What :meth:`ModelPipeline.compile <torq.lab.pipeline.ModelPipeline.compile>` produced.

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
    """What :meth:`ModelPipeline.run <torq.lab.pipeline.ModelPipeline.run>` produced.

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
