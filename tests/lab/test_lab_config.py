# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for layered config loading and the CLI --config wiring."""

import json

from torq.lab.cli.parser import _build_parser, _config_from_args, _remote_from_args
from torq.lab.pipeline.workflow import load_config


def test_load_config_later_source_wins():
    merged = load_config([
        {"model_path": "m.mlir", "chip": "SL2610", "compiler_options": ["--a"]},
        {"chip": "SL1620", "remote": {"address": "board"}},
    ])
    assert merged["model_path"] == "m.mlir"
    assert merged["chip"] == "SL1620"
    assert merged["remote"] == {"address": "board"}


def test_load_config_deep_merges_nested_dicts():
    merged = load_config([
        {"remote": {"address": "board", "port": 22}},
        {"remote": {"port": 2222}},
    ])
    assert merged["remote"] == {"address": "board", "port": 2222}


def test_load_config_reads_files_and_manifest_section(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps({"model_path": "m.mlir", "chip": "SL2610"}))
    # A full manifest source contributes only its "config" section.
    (tmp_path / "over.json").write_text(
        json.dumps({"schema_version": 2, "config": {"chip": "SL1620"}, "results": {}})
    )
    merged = load_config([tmp_path / "base.json", tmp_path / "over.json"])
    assert merged["model_path"] == "m.mlir"
    assert merged["chip"] == "SL1620"


def test_cli_config_builds_config_and_remote(tmp_path):
    cfg = tmp_path / "pipe.json"
    cfg.write_text(json.dumps({
        "model_path": str(tmp_path / "m.mlir"),
        "work_dir": str(tmp_path / "out"),
        "chip": "SL1620",
        "remote": {"address": "user@host", "port": 2222},
    }))
    args = _build_parser().parse_args(["run", "--config", str(cfg)])
    config = _config_from_args(args)
    assert config.model_path == tmp_path / "m.mlir"
    assert config.chip == "SL1620"
    remote = _remote_from_args(args)
    assert remote.address == "user@host"
    assert remote.port == 2222


def test_cli_positional_model_overrides_config(tmp_path):
    cfg = tmp_path / "pipe.json"
    cfg.write_text(json.dumps({"model_path": str(tmp_path / "base.mlir"), "work_dir": str(tmp_path / "out")}))
    args = _build_parser().parse_args(["compile", str(tmp_path / "override.mlir"), "--config", str(cfg)])
    assert _config_from_args(args).model_path == tmp_path / "override.mlir"


def test_cli_flags_override_config(tmp_path):
    # Explicit run-stage flags must win over --config values and fill unset ones
    # (decision 12); a config-only key the flags don't touch is preserved.
    cfg = tmp_path / "pipe.json"
    cfg.write_text(json.dumps({
        "model_path": str(tmp_path / "m.mlir"),
        "work_dir": str(tmp_path / "out"),
        "chip": "SL2610",
        "random_inputs": False,
    }))
    args = _build_parser().parse_args([
        "run", "--config", str(cfg), "--chip", "SL5", "--random-inputs", "--seed", "99",
    ])
    config = _config_from_args(args)
    assert config.chip == "SL5"          # flag overrides config
    assert config.random_inputs is True  # store_true flag overrides config
    assert config.input_seed == 99       # flag fills a key absent from config
    assert config.runtime_hw_type == "sim"  # unset flag falls back to the default


def test_cli_config_only_ignores_unset_flags(tmp_path):
    # Without the flags, config values stand (no spurious override to a default).
    cfg = tmp_path / "pipe.json"
    cfg.write_text(json.dumps({
        "model_path": str(tmp_path / "m.mlir"), "work_dir": str(tmp_path / "out"),
        "chip": "SL1620", "random_inputs": True, "input_seed": 7,
    }))
    args = _build_parser().parse_args(["run", "--config", str(cfg)])
    config = _config_from_args(args)
    assert config.chip == "SL1620"
    assert config.random_inputs is True
    assert config.input_seed == 7
