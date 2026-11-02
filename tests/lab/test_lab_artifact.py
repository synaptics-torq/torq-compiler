# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for torq.lab.pipeline.artifacts.describe()'s precedence resolution."""

import json

from torq.lab.pipeline.artifacts import describe

TOSA_MLIR = """
module {
  func.func @qkv(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
"""


def _write_json(path, data):
    path.write_text(json.dumps(data))


def test_tier1_explicit_manifest_wins_over_colocated(tmp_path):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKE")
    _write_json(vmfb.parent / "manifest.json", {
        "schema_version": 3, "config": {}, "results": {},
        "artifact": {"function": "colocated_fn", "chip": "COLOCATED"},
    })
    explicit = tmp_path / "explicit_manifest.json"
    _write_json(explicit, {
        "schema_version": 3, "config": {}, "results": {},
        "artifact": {"function": "explicit_fn", "chip": "EXPLICIT"},
    })

    info = describe(vmfb, manifest=explicit)

    assert info.function == "explicit_fn"
    assert info.chip == "EXPLICIT"
    assert info.manifest_path == explicit
    assert info.provenance["function"] == "explicit manifest"


def test_tier2_colocated_manifest_v3(tmp_path):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKE")
    _write_json(vmfb.parent / "manifest.json", {
        "schema_version": 3,
        "config": {},
        "results": {},
        "artifact": {
            "function": "qkv",
            "chip": "SL2610",
            "entry_points": ["qkv"],
            "io_spec": {
                "inputs": [{"shape": [1, 4], "fmt": "f32", "name": None}],
                "outputs": [{"shape": [1, 4], "fmt": "f32", "name": None}],
            },
            "source_kind": "mlir",
        },
    })

    info = describe(vmfb)

    assert info.function == "qkv"
    assert info.chip == "SL2610"
    assert info.entry_points == ["qkv"]
    assert info.io_spec.inputs[0].shape == [1, 4]
    assert info.source_kind == "mlir"
    assert info.provenance["function"] == "manifest.json"
    assert info.provenance["io_spec"] == "manifest.json"


def test_tier2_v2_manifest_without_artifact_section_falls_back(tmp_path):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKE")
    _write_json(vmfb.parent / "manifest.json", {
        "schema_version": 2,
        "config": {"chip": "SL1620", "function": "legacy_fn", "model_path": "m.mlir"},
        "results": {"compile": {"debug_dir": str(tmp_path / "debug"), "command": ["torq-compile"]}},
    })

    info = describe(vmfb)

    assert info.function == "legacy_fn"
    assert info.chip == "SL1620"
    assert info.debug_dir == tmp_path / "debug"
    assert info.compile_command == ["torq-compile"]


def test_tier3_sibling_mlir(tmp_path):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKE")
    (tmp_path / "model.mlir").write_text(TOSA_MLIR)

    info = describe(vmfb)

    assert info.function == "qkv"
    assert info.source_kind == "mlir"
    assert info.entry_points == ["qkv"]
    assert info.io_spec.inputs[0].fmt == "f32"
    assert info.provenance["source_path"] == "sibling model.mlir"
    assert info.provenance["function"] == "sibling model.mlir"


def test_no_data_guesses_main_and_records_provenance(tmp_path):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKE")

    info = describe(vmfb)

    assert info.function == "main"
    assert "guessed" in info.provenance["function"]
    assert info.io_spec is None


