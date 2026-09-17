# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Contracts for the unified ONNX quantize/analyze CLI (``torq-lab quantize`` /
``torq-lab analyze`` / ``torq-quantize-model``).

Static and dynamic quantization run in-process against a small generated
MatMul model, so the tests only need the ``[onnx]`` extra.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.lab.cli import commands as cli
from torq.lab.quantization.onnx import quantize_onnx_model
from torq.lab.quantization.onnx.static import is_model_quantized, onnx_static_quantize


def _make_matmul_model() -> onnx.ModelProto:
    weight = numpy_helper.from_array(np.eye(2, dtype=np.float32), "weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="MatMul_0")],
        "quantize-cli-matmul",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model


def _make_decoder_model(path) -> Path:
    """Minimal static-decoder ONNX graph for the weights analyze flow.

    One embedding-input MatMul layer plus a single KV-cache pass-through,
    so `LayerSensitivityAnalyzer` can auto-detect the architecture.
    """
    weight = numpy_helper.from_array(
        np.random.randn(8, 16).astype(np.float32), "weight"
    )
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["token_embedding", "weight"], ["logits"], name="MatMul_0"),
            helper.make_node("Identity", ["past_key_values.0.key_value"], ["present.0.key_value"], name="Identity_0"),
        ],
        "analyze-cli-decoder",
        [
            helper.make_tensor_value_info("token_embedding", TensorProto.FLOAT, [1, 1, 8]),
            helper.make_tensor_value_info("past_key_values.0.key_value", TensorProto.FLOAT, [1, 2, 4, 4]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 16]),
            helper.make_tensor_value_info("present.0.key_value", TensorProto.FLOAT, [1, 2, 4, 4]),
        ],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, str(path))
    return path


def _make_matmul_chain_model(path=None):
    """Two-MatMul chain (with a Relu in between) for node-selection tests."""
    w1 = numpy_helper.from_array(np.random.randn(4, 6).astype(np.float32), "w1")
    w2 = numpy_helper.from_array(np.random.randn(6, 2).astype(np.float32), "w2")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "w1"], ["h"], name="MatMul_0"),
            helper.make_node("Relu", ["h"], ["h_r"], name="Relu_0"),
            helper.make_node("MatMul", ["h_r", "w2"], ["output"], name="MatMul_1"),
        ],
        "quantize-cli-matmul-chain",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [w1, w2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    if path is not None:
        onnx.save(model, str(path))
        return path
    return model


def _weight_quantized(model: onnx.ModelProto, name: str) -> bool:
    """True if *name* (or its QDQ replacement ``<name>_quantized``) is int8/uint8."""
    dtypes = {i.name: i.data_type for i in model.graph.initializer}
    return any(
        dtypes.get(candidate) in (TensorProto.INT8, TensorProto.UINT8)
        for candidate in (f"{name}_quantized", name)
    )


def test_onnx_static_quantize_only_nodes():
    quantized = onnx_static_quantize(
        _make_matmul_chain_model(), num_calib=2, quantize_only_nodes=["MatMul_0"]
    )
    assert _weight_quantized(quantized, "w1")
    assert not _weight_quantized(quantized, "w2")


def test_onnx_static_quantize_exclude_nodes():
    quantized = onnx_static_quantize(
        _make_matmul_chain_model(), num_calib=2, exclude_nodes=["MatMul_0"]
    )
    assert not _weight_quantized(quantized, "w1")
    assert _weight_quantized(quantized, "w2")


def test_onnx_static_quantize_only_ops():
    quantized = onnx_static_quantize(
        _make_matmul_chain_model(), num_calib=2, quantize_only_ops=["Relu"]
    )
    assert not _weight_quantized(quantized, "w1")
    assert not _weight_quantized(quantized, "w2")


def test_onnx_static_quantize_rejects_unknown_node_names():
    # unknown --quantize-only-nodes / --exclude-nodes values used to be
    # ignored by the ORT quantizer, silently producing an unquantized model.
    with pytest.raises(ValueError, match="unknown node name"):
        onnx_static_quantize(
            _make_matmul_chain_model(), num_calib=2, quantize_only_nodes=["DoesNotExist"]
        )
    with pytest.raises(ValueError, match="unknown node name"):
        onnx_static_quantize(
            _make_matmul_chain_model(), num_calib=2, exclude_nodes=["Nope"]
        )


def test_onnx_static_quantize_rejects_unknown_op_types():
    with pytest.raises(ValueError, match="not present in the model"):
        onnx_static_quantize(
            _make_matmul_chain_model(), num_calib=2, quantize_only_ops=["NotAnOp"]
        )


def test_onnx_static_quantize_rejects_already_quantized():
    # re-quantizing a quantized model used to no-op silently; analyze
    # static refuses, so quantize static must too.
    quantized = onnx_static_quantize(_make_matmul_chain_model(), num_calib=2)
    assert is_model_quantized(quantized)
    with pytest.raises(ValueError, match="already quantized"):
        onnx_static_quantize(quantized, num_calib=2)


def test_quantize_cli_static_rejects_unknown_nodes(model_path, tmp_path):
    rc = cli.main(
        ["quantize", "static", "-i", str(model_path), "--quantize-only-nodes", "DoesNotExist"]
    )
    assert rc == 1


def test_quantize_cli_static_rejects_already_quantized(model_path, tmp_path):
    quantized = tmp_path / "model_int8.onnx"
    assert cli.main(["quantize", "static", "-i", str(model_path), "-o", str(quantized)]) == 0
    rc = cli.main(
        ["quantize", "static", "-i", str(quantized), "-o", str(tmp_path / "again.onnx")]
    )
    assert rc == 1
    assert not (tmp_path / "again.onnx").exists()


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(_make_matmul_model(), path)
    return path


def test_quantize_cli_static(model_path, tmp_path):
    output = tmp_path / "model_static.onnx"
    rc = cli.main(["quantize", "static", "-i", str(model_path), "-o", str(output)])
    assert rc == 0
    assert output.exists()
    assert is_model_quantized(onnx.load(str(output)))


def test_quantize_cli_static_default_output(model_path):
    rc = cli.main(["quantize", "static", "-i", str(model_path)])
    assert rc == 0
    assert model_path.with_suffix(".int8.onnx").exists()


def test_quantize_cli_static_missing_input(tmp_path):
    rc = cli.main(["quantize", "static", "-i", str(tmp_path / "nope.onnx")])
    assert rc == 1


def test_quantize_cli_static_quantize_only_nodes(tmp_path):
    model_path = _make_matmul_chain_model(tmp_path / "chain.onnx")
    output = tmp_path / "chain_static.onnx"
    rc = cli.main(
        [
            "quantize", "static",
            "-i", str(model_path), "-o", str(output),
            "--quantize-only-nodes", "MatMul_0",
        ]
    )
    assert rc == 0
    quantized = onnx.load(str(output))
    assert _weight_quantized(quantized, "w1")
    assert not _weight_quantized(quantized, "w2")


def test_quantize_cli_static_exclude_nodes(tmp_path):
    model_path = _make_matmul_chain_model(tmp_path / "chain.onnx")
    output = tmp_path / "chain_static.onnx"
    rc = cli.main(
        [
            "quantize", "static",
            "-i", str(model_path), "-o", str(output),
            "--exclude-nodes", "MatMul_0",
        ]
    )
    assert rc == 0
    quantized = onnx.load(str(output))
    assert not _weight_quantized(quantized, "w1")
    assert _weight_quantized(quantized, "w2")


def test_quantize_cli_dynamic_missing_input(tmp_path):
    rc = cli.main(
        ["quantize", "dynamic", "-i", str(tmp_path / "nope.onnx"), "-o", str(tmp_path / "o.onnx")]
    )
    assert rc == 1


def test_quantize_cli_weights_missing_input(tmp_path):
    rc = cli.main(
        ["quantize", "weights", "-i", str(tmp_path / "nope.onnx"), "-o", str(tmp_path / "o.onnx"), "--bits", "8"]
    )
    assert rc == 1


def test_quantize_cli_dynamic(model_path, tmp_path):
    output = tmp_path / "model_dynamic.onnx"
    rc = cli.main(["quantize", "dynamic", "-i", str(model_path), "-o", str(output)])
    assert rc == 0
    assert output.exists()
    assert is_model_quantized(onnx.load(str(output)))


def test_quantize_cli_weights(model_path, tmp_path):
    output = tmp_path / "model_weights.onnx"
    rc = cli.main(
        ["quantize", "weights", "-i", str(model_path), "-o", str(output), "--bits", "8"]
    )
    assert rc == 0
    assert output.exists()
    model = onnx.load(str(output))
    assert any(n.op_type == "DequantizeLinear" for n in model.graph.node)


def test_quantize_cli_weights_requires_bits_or_config(model_path, tmp_path):
    rc = cli.main(
        ["quantize", "weights", "-i", str(model_path), "-o", str(tmp_path / "w.onnx")]
    )
    assert rc == 1


def test_analyze_cli_dynamic(model_path, tmp_path):
    report = tmp_path / "sensitivity.json"
    rc = cli.main(["analyze", "dynamic", "-i", str(model_path), "-o", str(report)])
    assert rc == 0
    results = json.loads(report.read_text())
    assert len(results) == 1
    entry = results[0]
    assert entry["node"]
    assert entry["op_type"] == "MatMul"
    assert {"kl", "cosine", "max_abs_error", "classification"} <= entry.keys()


def test_analyze_cli_static(model_path, tmp_path):
    report = tmp_path / "sensitivity.json"
    rc = cli.main(["analyze", "static", "-i", str(model_path), "-o", str(report)])
    assert rc == 0
    results = json.loads(report.read_text())
    assert len(results) == 1
    entry = results[0]
    assert entry["node"] == "MatMul_0"
    assert entry["op_type"] == "MatMul"
    assert {"kl", "cosine", "max_abs_error", "classification"} <= entry.keys()


def test_analyze_cli_static_already_quantized(model_path, tmp_path):
    quantized = tmp_path / "model_int8.onnx"
    assert cli.main(["quantize", "static", "-i", str(model_path), "-o", str(quantized)]) == 0
    rc = cli.main(["analyze", "static", "-i", str(quantized), "-o", str(tmp_path / "s.json")])
    assert rc == 1


def test_analyze_cli_static_exclude_roundtrip(tmp_path):
    """An analysis report feeds back into `quantize static --exclude-nodes`."""
    model_path = _make_matmul_chain_model(tmp_path / "chain.onnx")
    report = tmp_path / "sensitivity.json"
    exclude = tmp_path / "exclude.json"
    rc = cli.main(
        [
            "analyze", "static",
            "-i", str(model_path), "-o", str(report),
            "--exclude-output", str(exclude), "--exclude-class", "MEDIUM",
        ]
    )
    assert rc == 0
    results = json.loads(report.read_text())
    excluded = json.loads(exclude.read_text())
    assert set(excluded) <= {r["node"] for r in results}
    # Excluding every analyzed node keeps all MatMul weights unquantized.
    output = tmp_path / "chain_static.onnx"
    rc = cli.main(
        [
            "quantize", "static",
            "-i", str(model_path), "-o", str(output),
            "--exclude-nodes", *[r["node"] for r in results],
        ]
    )
    assert rc == 0
    quantized = onnx.load(str(output))
    assert not _weight_quantized(quantized, "w1")
    assert not _weight_quantized(quantized, "w2")


def test_write_exclude_list_thresholds(tmp_path):
    from torq.lab.quantization.onnx._analysis import write_exclude_list

    results = [
        {"node": "a", "classification": "LOW"},
        {"node": "b", "classification": "HIGH"},
        {"node": "c", "classification": "CRITICAL"},
    ]
    path = tmp_path / "exclude.json"
    assert write_exclude_list(results, path, "HIGH") == 2
    assert json.loads(path.read_text()) == ["b", "c"]


def test_analyze_cli_weights(tmp_path):
    model_path = _make_decoder_model(tmp_path / "model.onnx")
    embeddings = tmp_path / "token_embeddings.npy"
    np.save(embeddings, np.random.randn(16, 8).astype(np.float32))
    pre_tokenized = tmp_path / "pre_tokenized.json"
    pre_tokenized.write_text(json.dumps([[1, 2, 3]]))
    report = tmp_path / "sensitivity.json"
    config = tmp_path / "quant_config.json"
    rc = cli.main(
        [
            "analyze", "weights",
            "-i", str(model_path), "-o", str(report),
            "--config-output", str(config),
            "--embeddings", str(embeddings),
            "--pre-tokenized-file", str(pre_tokenized),
            "--num-tokens", "3",
        ]
    )
    assert rc == 0
    results = json.loads(report.read_text())
    assert len(results) == 1
    entry = results[0]
    assert entry["layer_name"] == "MatMul_0"
    assert set(entry["kl_divergence"]) == {"4", "8", "16"}
    assert {"cosine_similarity", "top1_match", "classification"} <= entry.keys()
    cfg = json.loads(config.read_text())
    assert "default" in cfg and "layers" in cfg


def test_quantize_model_cli_analyze_weights(tmp_path):
    """torq-quantize-model carries the analyze weights command too."""
    from torq.lab.quantization.onnx import cli as quantize_cli

    model_path = _make_decoder_model(tmp_path / "model.onnx")
    embeddings = tmp_path / "token_embeddings.npy"
    np.save(embeddings, np.random.randn(16, 8).astype(np.float32))
    pre_tokenized = tmp_path / "pre_tokenized.json"
    pre_tokenized.write_text(json.dumps([[1, 2, 3]]))
    report = tmp_path / "sensitivity.json"
    rc = quantize_cli.main(
        [
            "analyze", "weights",
            "-i", str(model_path), "-o", str(report),
            "--embeddings", str(embeddings),
            "--pre-tokenized-file", str(pre_tokenized),
            "--num-tokens", "3",
            "--bits", "8", "16",
        ]
    )
    assert rc == 0
    results = json.loads(report.read_text())
    assert len(results) == 1
    assert set(results[0]["kl_divergence"]) == {"8", "16"}


def test_quantize_model_cli_analyze_dynamic(model_path, tmp_path):
    """torq-quantize-model carries the analyze command tree too."""
    from torq.lab.quantization.onnx import cli as quantize_cli

    report = tmp_path / "sensitivity.json"
    rc = quantize_cli.main(["analyze", "dynamic", "-i", str(model_path), "-o", str(report)])
    assert rc == 0
    results = json.loads(report.read_text())
    assert len(results) == 1
    assert results[0]["op_type"] == "MatMul"


def test_quantize_model_cli_analyze_static(model_path, tmp_path):
    """torq-quantize-model carries the analyze static command too."""
    from torq.lab.quantization.onnx import cli as quantize_cli

    report = tmp_path / "sensitivity.json"
    rc = quantize_cli.main(["analyze", "static", "-i", str(model_path), "-o", str(report)])
    assert rc == 0
    results = json.loads(report.read_text())
    assert len(results) == 1
    assert results[0]["node"] == "MatMul_0"


def test_quantize_cli_helpers_live_in_mode_modules():
    """Each quantization mode exposes its quantize/analyze CLI helpers from its own package."""
    from torq.lab.quantization.onnx import dynamic, static, weights

    assert callable(dynamic.add_dynamic_quantize_args)
    assert callable(dynamic.dynamic_quantize_from_args)
    assert callable(dynamic.add_dynamic_analyze_args)
    assert callable(dynamic.dynamic_analyze_from_args)
    assert callable(static.add_static_quantize_args)
    assert callable(static.static_quantize_from_args)
    assert callable(static.add_static_analyze_args)
    assert callable(static.static_analyze_from_args)
    assert callable(weights.add_weights_quantize_args)
    assert callable(weights.weights_quantize_from_args)
    assert callable(weights.add_weights_analyze_args)
    assert callable(weights.weights_analyze_from_args)


def test_quantize_cli_help_lists_formats(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["quantize", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "torq-lab quantize" in out
    assert "static" in out and "dynamic" in out and "weights" in out


def test_analyze_cli_help_lists_formats(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["analyze", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "torq-lab analyze" in out
    assert "static" in out and "dynamic" in out and "weights" in out


def test_quantize_model_entry_prog(capsys):
    from torq.lab.quantization.onnx import cli as quantize_cli

    with pytest.raises(SystemExit) as exc:
        quantize_cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "torq-quantize-model" in out
    assert "static" in out and "dynamic" in out and "weights" in out and "analyze" in out


def test_quantize_cli_orchestrator_weights(model_path, tmp_path):
    output = tmp_path / "w.onnx"
    quantize_onnx_model(model_path, output, method="weights", bits=8)
    assert output.exists()
    model = onnx.load(str(output))
    assert any(n.op_type == "DequantizeLinear" for n in model.graph.node)


def test_quantize_cli_orchestrator_unknown_method(model_path, tmp_path):
    with pytest.raises(ValueError, match="unknown ONNX quantization method"):
        quantize_onnx_model(model_path, tmp_path / "x.onnx", method="bogus")
