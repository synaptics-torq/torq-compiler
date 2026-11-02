# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for torq.lab.cli.output's Plan/Summary dataclasses and formatters."""

from pathlib import Path

from torq.lab.pipeline.artifacts import ArtifactInfo
from torq.lab.cli.output import (
    Plan,
    StageTime,
    Summary,
    format_inspect,
    format_plan,
    format_summary,
)


def test_format_plan_names_import_and_compile_outputs():
    plan = Plan(
        command="run", model_path=Path("m.onnx"), import_output=Path("w/m.mlir"),
        compile_output=Path("w/m.vmfb"), execution_target="sim",
        input_source="random (seed=1234)", artifacts_dir=Path("w"), touches_board=False,
    )
    text = format_plan(plan)
    assert "import  -> w/m.mlir" in text
    assert "compile -> w/m.vmfb" in text
    assert "random (seed=1234)" in text
    assert "touches a board" not in text


def test_summary_to_dict_and_format():
    summary = Summary(
        command="run", status="ok",
        stages=[StageTime("compile", 1.5), StageTime("run", 0.25)],
        target="sim", input_provenance="random (seed=1234)",
        profile_quality="raw", reference=None, next_command="torq-lab inspect m.vmfb",
    )
    d = summary.to_dict()
    assert d["stages"] == [{"name": "compile", "elapsed_s": 1.5}, {"name": "run", "elapsed_s": 0.25}]
    text = format_summary(summary)
    assert "compile: 1.50s" in text
    assert "run: 0.25s" in text
    assert "profile: raw" in text
    assert "next: torq-lab inspect m.vmfb" in text


def test_format_inspect_reports_facts_and_provenance():
    info = ArtifactInfo(
        vmfb_path=Path("m.vmfb"), source_path=Path("m.mlir"), source_kind="mlir",
        function="qkv", entry_points=["qkv"], chip="SL2610",
        provenance={"function": "sibling m.mlir"},
    )
    text = format_inspect(info, ["torq-lab run m.vmfb"], annotated_profile_possible=False)
    assert "Function: qkv [sibling m.mlir]" in text
    assert "Chip: SL2610" in text
    assert "Annotated profiling possible: no" in text
    assert "torq-lab run m.vmfb" in text
