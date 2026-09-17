# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan, summary, and inspect view models and their text/JSON renderers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _s(p):
    return None if p is None else str(p)


@dataclass
class Plan:
    """What a command would do, computed with no import, compile, run, or network access."""

    command: str
    model_path: Path
    import_output: Optional[Path] = None
    compile_output: Optional[Path] = None
    execution_target: str = ""
    input_source: str = ""
    artifacts_dir: Optional[Path] = None
    touches_board: bool = False

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "model_path": _s(self.model_path),
            "import_output": _s(self.import_output),
            "compile_output": _s(self.compile_output),
            "execution_target": self.execution_target,
            "input_source": self.input_source,
            "artifacts_dir": _s(self.artifacts_dir),
            "touches_board": self.touches_board,
        }


@dataclass
class StageTime:
    """Elapsed wall-clock time for one pipeline stage (``"compile"``, ``"run"``, ...)."""

    name: str
    elapsed: float


@dataclass
class Summary:
    """What a command produced: per-stage timing, target, inputs, and what to do next."""

    command: str
    status: str
    stages: List[StageTime] = field(default_factory=list)
    target: str = ""
    input_provenance: str = ""
    profile_quality: Optional[str] = None
    reference: Optional[str] = None
    next_command: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "status": self.status,
            "stages": [{"name": s.name, "elapsed_s": s.elapsed} for s in self.stages],
            "target": self.target,
            "input_provenance": self.input_provenance,
            "profile_quality": self.profile_quality,
            "reference": self.reference,
            "next_command": self.next_command,
        }


def format_plan(plan: Plan) -> str:
    """Render a :class:`Plan` as copy-pasteable human text."""
    lines = [f"Plan: {plan.command} {plan.model_path}"]
    if plan.import_output:
        lines.append(f"  import  -> {plan.import_output}")
    if plan.compile_output:
        lines.append(f"  compile -> {plan.compile_output}")
    board = " (touches a board)" if plan.touches_board else ""
    lines.append(f"  execute on {plan.execution_target}{board}")
    lines.append(f"  inputs: {plan.input_source}")
    if plan.artifacts_dir:
        lines.append(f"  artifacts: {plan.artifacts_dir}")
    return "\n".join(lines)


def format_summary(summary: Summary) -> str:
    """Render a :class:`Summary` as human text."""
    lines = [f"{summary.command}: {summary.status}"]
    for stage in summary.stages:
        lines.append(f"  {stage.name}: {stage.elapsed:.2f}s")
    lines.append(f"  target: {summary.target}")
    lines.append(f"  inputs: {summary.input_provenance}")
    if summary.profile_quality:
        lines.append(f"  profile: {summary.profile_quality}")
    if summary.reference:
        lines.append(f"  reference: {summary.reference}")
    if summary.next_command:
        lines.append(f"  next: {summary.next_command}")
    return "\n".join(lines)


def format_inspect(info, hints: List[str], annotated_profile_possible: bool) -> str:
    """Render an :class:`~torq.lab.pipeline.artifacts.ArtifactInfo` (``inspect``'s only output shape)."""
    lines = []
    if info.vmfb_path:
        lines.append(f"VMFB: {info.vmfb_path}")
    if info.source_path:
        lines.append(f"Source: {info.source_path} ({info.source_kind or 'unknown kind'})")
    lines.append(f"Function: {info.function} [{info.provenance.get('function', 'unknown')}]")
    if info.entry_points:
        lines.append(f"Entry points: {', '.join(info.entry_points)} [{info.provenance.get('entry_points', 'unknown')}]")
    if info.io_spec:
        source = info.provenance.get("io_spec", "unknown")
        lines.append(f"Inputs: {[t.to_arg() for t in info.io_spec.inputs]} [{source}]")
        lines.append(f"Outputs: {[t.to_arg() for t in info.io_spec.outputs]} [{source}]")
    else:
        lines.append("Inputs/Outputs: unknown")
    if info.converted_io:
        lines.append(f"Converted I/O: {info.converted_io}")
    lines.append(f"Chip: {info.chip or 'unknown'}")
    lines.append(f"Debug info: {'yes' if info.debug_dir else 'no'}")
    lines.append(f"Annotated profiling possible: {'yes' if annotated_profile_possible else 'no'}")
    if hints:
        lines.append("Next:")
        for hint in hints:
            lines.append(f"  {hint}")
    return "\n".join(lines)
