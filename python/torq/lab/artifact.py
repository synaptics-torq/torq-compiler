# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve artifact metadata (function, I/O spec, provenance) for a VMFB/model."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from torq.lab import io
from torq.lab.types import MlirIoSpec, TensorType

_ENTRY_POINT_RE = re.compile(r"^\s*func\.func.*")


def _s(p):
    return None if p is None else str(p)


def _p(v):
    return Path(v) if v is not None else None


@dataclass
class ArtifactInfo:
    """Resolved facts about a VMFB (or a not-yet-compiled model), with provenance."""

    vmfb_path: Optional[Path] = None
    source_path: Optional[Path] = None  # the .onnx or .mlir the VMFB came from
    source_kind: Optional[str] = None  # "onnx" | "mlir" | None
    manifest_path: Optional[Path] = None
    function: Optional[str] = None
    entry_points: List[str] = field(default_factory=list)
    io_spec: Optional[MlirIoSpec] = None  # public signature, post-conversion when recorded in a manifest
    converted_io: List[str] = field(default_factory=list)  # the effective convert-io policy
    debug_dir: Optional[Path] = None
    chip: Optional[str] = None
    compile_command: List[str] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict:
        """Serialize the recorded facts for the manifest's ``artifact`` section."""
        return {
            "source_path": _s(self.source_path),
            "source_kind": self.source_kind,
            "function": self.function,
            "entry_points": list(self.entry_points),
            "io_spec": _io_spec_to_dict(self.io_spec),
            "converted_io": list(self.converted_io),
            "debug_dir": _s(self.debug_dir),
            "chip": self.chip,
            "compile_command": list(self.compile_command),
        }


def _io_spec_to_dict(spec: Optional[MlirIoSpec]) -> Optional[dict]:
    if spec is None:
        return None

    def tensors(ts):
        return [{"shape": list(t.shape), "fmt": t.fmt, "name": t.name} for t in ts]

    return {"inputs": tensors(spec.inputs), "outputs": tensors(spec.outputs)}


def _io_spec_from_dict(d: Optional[dict]) -> Optional[MlirIoSpec]:
    if not d:
        return None

    def tensors(ts):
        return [TensorType(list(t["shape"]), t["fmt"], t.get("name")) for t in ts]

    return MlirIoSpec(inputs=tensors(d.get("inputs", [])), outputs=tensors(d.get("outputs", [])))


def _entry_points_from_mlir(mlir_path: Path) -> List[str]:
    names = []
    for line in Path(mlir_path).read_text().split("\n"):
        if _ENTRY_POINT_RE.match(line):
            m = re.search(r"@(\w+)\s*\(", line) or re.search(r'@"([^"]+)"\s*\(', line)
            if m:
                names.append(m.group(1))
    return names


def _find_sibling_source(vmfb_path: Path) -> Optional[Path]:
    for suffix in (".mlir", ".onnx"):
        candidate = vmfb_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _load_manifest(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _apply_manifest(manifest: dict, manifest_path: Path, info: ArtifactInfo, label: str) -> None:
    """Fill unresolved ``info`` fields from a loaded manifest dict, recording provenance."""
    config = manifest.get("config", {})
    results = manifest.get("results", {})
    compile_results = results.get("compile", {})
    artifact = manifest.get("artifact")  # only present from schema_version >= 3

    def _set(name, value):
        if value is not None and getattr(info, name) is None:
            setattr(info, name, value)
            info.provenance[name] = label

    def _set_list(name, value):
        if value and not getattr(info, name):
            setattr(info, name, list(value))
            info.provenance[name] = label

    if artifact:
        _set("function", artifact.get("function"))
        _set_list("entry_points", artifact.get("entry_points"))
        if artifact.get("io_spec") and info.io_spec is None:
            info.io_spec = _io_spec_from_dict(artifact["io_spec"])
            info.provenance["io_spec"] = label
        _set_list("converted_io", artifact.get("converted_io"))
        _set("debug_dir", _p(artifact.get("debug_dir")))
        _set("chip", artifact.get("chip"))
        _set_list("compile_command", artifact.get("compile_command"))
        _set("source_path", _p(artifact.get("source_path")))
        _set("source_kind", artifact.get("source_kind"))
    else:
        # v1/v2 manifest predating the artifact section: fall back to the
        # config/results fields that already carried these facts.
        _set("function", config.get("function"))
        _set("chip", config.get("chip"))
        _set("debug_dir", _p(compile_results.get("debug_dir")))
        _set_list("compile_command", compile_results.get("command"))
        _set("source_path", _p(config.get("spec_source")) or _p(config.get("model_path")))

    if info.vmfb_path is None and compile_results.get("vmfb_path"):
        info.vmfb_path = Path(compile_results["vmfb_path"])
        info.provenance["vmfb_path"] = label

    info.manifest_path = manifest_path


def describe(path, *, manifest=None, source=None, input_specs=None, output_specs=None) -> ArtifactInfo:
    """Resolve an artifact's metadata by precedence, recording where each fact came from.

    ``path`` is a ``.vmfb``, a ``.mlir``/``.onnx`` source, or an artifact
    directory. Precedence: (1) ``manifest``, an explicitly supplied manifest
    path; (2) a colocated ``manifest.json`` next to ``path``; (3) a sibling
    ``.mlir``/``.onnx`` next to a VMFB, or the caller's ``source`` override
    (e.g. ``PipelineConfig.spec_source``); (4) MLIR-derived I/O spec;
    (5) explicit input/output specs from ``input_specs``/``output_specs``.
    A function that still can't be resolved after every tier falls back to
    ``"main"`` and recorded as a guess in :attr:`ArtifactInfo.provenance`.
    """
    path = Path(path)
    info = ArtifactInfo()
    colocated_manifest: Optional[Path] = None

    if path.is_dir():
        colocated_manifest = path / "manifest.json"
        vmfbs = sorted(path.glob("*.vmfb"))
        if len(vmfbs) == 1:
            info.vmfb_path = vmfbs[0]
    elif path.suffix == ".vmfb":
        info.vmfb_path = path
        colocated_manifest = path.parent / "manifest.json"
    elif path.suffix in (".mlir", ".onnx"):
        info.source_path = path
        info.source_kind = path.suffix.lstrip(".")
        colocated_manifest = path.parent / "manifest.json"

    # Tier 1: an explicitly supplied manifest.
    if manifest is not None:
        manifest_path = Path(manifest)
        data = _load_manifest(manifest_path)
        if data is not None:
            _apply_manifest(data, manifest_path, info, "explicit manifest")

    # Tier 2: a colocated manifest.json.
    if info.manifest_path is None and colocated_manifest is not None and colocated_manifest.exists():
        data = _load_manifest(colocated_manifest)
        if data is not None:
            _apply_manifest(data, colocated_manifest, info, "manifest.json")

    # Tier 3: a sibling source file, or the caller's explicit override.
    if info.source_path is None:
        if source is not None:
            info.source_path = Path(source)
            info.provenance["source_path"] = "config.spec_source"
        elif info.vmfb_path is not None:
            sibling = _find_sibling_source(info.vmfb_path)
            if sibling is not None:
                info.source_path = sibling
                info.provenance["source_path"] = f"sibling {sibling.name}"

    if info.source_path is not None and info.source_kind is None:
        info.source_kind = info.source_path.suffix.lstrip(".")
        info.provenance.setdefault("source_kind", info.provenance.get("source_path"))

    if info.source_path is not None and info.source_path.suffix == ".mlir" and info.source_path.exists():
        label = info.provenance.get("source_path", str(info.source_path))
        if info.function is None:
            info.function = io.parse_func_name(info.source_path)
            info.provenance["function"] = label
        if not info.entry_points:
            info.entry_points = _entry_points_from_mlir(info.source_path)
            info.provenance["entry_points"] = label
        if info.io_spec is None:
            info.io_spec = io.parse_mlir_io_spec(info.source_path)
            info.provenance["io_spec"] = label

    # Tier 4: VMFB reflection is best-effort and optional; no reflection
    # binding is wired up yet, so this tier is currently a no-op rather than
    # a hard dependency.

    # Tier 5: explicit input/output specs supplied by the caller. Only fires
    # when at least one side is non-empty: empty lists (the CLI default) are not
    # a "supplied" spec, so an orphan VMFB keeps io_spec=None and the run path
    # can fail with recovery hints instead of fabricating a zero-input signature.
    if info.io_spec is None and (input_specs or output_specs):
        inputs = [TensorType.from_string(spec) for spec in (input_specs or [])]
        outputs = [TensorType.from_string(spec) for spec in (output_specs or [])]
        info.io_spec = MlirIoSpec(inputs=inputs, outputs=outputs)
        info.provenance["io_spec"] = "explicit input_specs/output_specs"

    if info.function is None:
        info.function = "main"
        info.provenance["function"] = "guessed default 'main' (no manifest, source, or reflection available)"

    return info
