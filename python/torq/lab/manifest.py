# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build and write the run manifest (``manifest.json``)."""

import json
import os
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 3


def _s(value):
    return None if value is None else str(value)


def build_manifest(
    *,
    config,
    compile_result=None,
    run_result=None,
    remote=None,
    comparison=None,
    artifact=None,
    status: str = "ok",
    diagnostics: str = "",
) -> dict:
    """Assemble a stable, JSON-serializable manifest describing a pipeline run.

    The manifest keeps two concerns separate: ``config`` (the pipeline inputs,
    round-trippable via ``PipelineConfig.from_manifest`` / ``RemoteTarget``) and
    ``results`` (what the run produced). This lets a pipeline be re-set-up from a
    saved manifest.

    ``artifact`` is an optional ``torq.lab.artifact.ArtifactInfo`` describing
    what the compiled VMFB is (entry function, I/O signature, source
    provenance); when given, it is stored verbatim () under the top-level 
    ``artifact`` key so a later ``artifact.describe()`` on this manifest can 
    resolve those facts without re-deriving them.
    """
    config_section = config.to_dict()
    if remote is not None:
        config_section["remote"] = remote.to_dict()

    results = {}
    if compile_result is not None:
        results["compile"] = {
            "vmfb_path": _s(compile_result.vmfb_path),
            "command": [str(c) for c in compile_result.command],
            "tool": _s(compile_result.command[0]) if compile_result.command else None,
            "elapsed_s": compile_result.elapsed,
            "debug_dir": _s(compile_result.debug_dir),
            "phases_dir": _s(compile_result.phases_dir),
            "compile_profile": _s(compile_result.compile_profile),
            "compile_trace": _s(compile_result.compile_trace),
            "perfetto_viewer": _s(compile_result.perfetto_viewer),
        }

    if run_result is not None:
        results["run"] = {
            "command": [str(c) for c in run_result.command],
            "tool": _s(run_result.command[0]) if run_result.command else None,
            "output_paths": [_s(p) for p in run_result.output_paths],
            "host_profile": _s(run_result.host_profile),
            "annotated_profile": _s(run_result.annotated_profile),
            "trace": _s(run_result.trace),
            "perfetto_viewer": _s(run_result.perfetto_viewer),
            "wall_time_s": run_result.wall_time,
        }

    if comparison is not None:
        results["comparison"] = {
            "passed": comparison.passed,
            "reason": comparison.reason,
        }

    results["diagnostics"] = diagnostics

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "config": config_section,
        "results": results,
    }
    if artifact is not None:
        manifest["artifact"] = artifact.to_manifest_dict()
    return manifest


def atomic_write_json_file(key_dir, filename: str, data, sort_keys: bool = False) -> None:
    """Atomically write ``filename`` as JSON inside ``key_dir``.

    Writes to ``<filename>.tmp.<pid>`` then renames it into place. POSIX
    ``rename`` is atomic, so a concurrent reader sees either the old file or the
    new one, never a partially written file. The caller is responsible for any
    locking needed to serialize writers of the same file.
    """
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = key_dir / f"{filename}.tmp.{os.getpid()}"
    tmp_path.unlink(missing_ok=True)
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=sort_keys))
    tmp_path.rename(key_dir / filename)


def atomic_write_json_manifest(key_dir, manifest: dict) -> None:
    """Atomically write ``manifest.json`` inside ``key_dir``."""
    atomic_write_json_file(key_dir, "manifest.json", manifest)


def write_manifest(work_dir, manifest: dict) -> Path:
    """Atomically write ``manifest.json`` into ``work_dir`` and return its path."""
    work_dir = Path(work_dir)
    atomic_write_json_file(work_dir, "manifest.json", manifest, sort_keys=True)
    return work_dir / "manifest.json"
