# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Behavioral contract tests for the ``torq-lab`` CLI.

Every case runs the CLI in a subprocess so the test process does not import the
implementation modules.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from _lab_fake_tools import write_fake_compile, write_fake_run

TESTS_DIR = Path(__file__).parent


def _run_lab(args, *, env=None):
    """Run ``python -m torq.lab <args>`` from a neutral cwd with stdin closed."""
    return subprocess.run(
        [sys.executable, "-m", "torq.lab", *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=TESTS_DIR,
        env=env,
    )


def test_unknown_subcommand_exits_2_on_stderr():
    result = _run_lab(["bogus"])
    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage: torq-lab" in result.stderr
    assert "invalid choice: 'bogus'" in result.stderr


def test_unknown_option_exits_2_on_stderr():
    result = _run_lab(["compile", "--bogus-flag", "m.mlir"])
    assert result.returncode == 2
    assert result.stdout == ""
    assert "unrecognized arguments: --bogus-flag" in result.stderr


def test_missing_model_exits_1_on_stderr():
    result = _run_lab(["compile"])
    assert result.returncode == 1
    assert result.stdout == ""
    assert "error: 'model' is required unless --config is given" in result.stderr


def test_convert_dtype_invalid_choice_exits_2():
    result = _run_lab(["convert_dtype", "bogus"])
    assert result.returncode == 2
    assert result.stdout == ""
    assert "invalid choice: 'bogus'" in result.stderr


def test_convert_static_invalid_choice_exits_2():
    result = _run_lab(["convert_static", "bogus"])
    assert result.returncode == 2
    assert result.stdout == ""
    assert "invalid choice: 'bogus'" in result.stderr


def test_no_arguments_launches_interactive_session():
    result = _run_lab([])
    assert result.returncode == 0
    assert "torq-lab interactive mode: press Ctrl-D to quit at any time." in result.stdout
    assert "Model file (.onnx/.tflite/.mlir/.vmfb):" in result.stdout


def test_json_summary_and_manifest_contract(tmp_path):
    model = tmp_path / "model.mlir"
    model.write_text(
        "module {\n"
        "  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {\n"
        "    return %arg0 : tensor<1x4xf32>\n"
        "  }\n"
        "}\n"
    )
    work_dir = tmp_path / "out"
    compile_tool = write_fake_compile(tmp_path / "tools")
    run_tool = write_fake_run(tmp_path / "tools")
    env = os.environ.copy()
    env.update(
        {
            "TORQ_COMPILE": str(compile_tool),
            "TORQ_RUN_MODULE": str(run_tool),
            "FAKE_OUTPUT_SIZES": str(1 * 4 * 4),
        }
    )

    result = _run_lab(
        [
            "run",
            str(model),
            "--work-dir",
            str(work_dir),
            "--random-inputs",
            "--json",
        ],
        env=env,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert set(summary) == {
        "command",
        "status",
        "stages",
        "target",
        "input_provenance",
        "profile_quality",
        "reference",
        "next_command",
    }
    assert summary["command"] == "run"
    assert summary["status"] == "ok"
    assert [stage["name"] for stage in summary["stages"]] == ["compile", "run"]
    assert all(isinstance(stage["elapsed_s"], (int, float)) for stage in summary["stages"])
    assert summary["target"] == "sim"
    assert summary["input_provenance"] == "random (seed=1234)"
    assert summary["profile_quality"] is None
    assert summary["reference"] is None
    assert summary["next_command"] == f"torq-lab inspect {work_dir / 'model.vmfb'}"

    manifest = json.loads((work_dir / "manifest.json").read_text())
    assert set(manifest) == {"schema_version", "status", "config", "results", "artifact"}
    assert manifest["schema_version"] == 3
    assert manifest["status"] == "ok"
    assert manifest["config"]["model_path"] == str(model)
    assert manifest["config"]["work_dir"] == str(work_dir)
    assert manifest["config"]["random_inputs"] is True
    assert set(manifest["results"]) == {"compile", "run", "diagnostics"}
    assert manifest["results"]["compile"]["vmfb_path"] == str(work_dir / "model.vmfb")
    assert manifest["results"]["run"]["output_paths"] == [
        str(work_dir / "outputs" / "output_0.bin")
    ]
    assert manifest["artifact"]["source_kind"] == "mlir"
