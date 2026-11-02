# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Functional tests for the TFLite dynamic-to-static shape conversion command."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from torq.lab.model_tools.shape_conversion.tflite import convert_model, main


def _model_t(model_path):
    from tensorflow.lite.python import schema_py_generated

    return schema_py_generated.ModelT.InitFromPackedBuf(bytearray(model_path.read_bytes()), 0)


def _shape_signatures(model_path):
    signatures = {}
    for subgraph in _model_t(model_path).subgraphs:
        for tensor in subgraph.tensors:
            if tensor.shapeSignature is not None:
                name = tensor.name
                if isinstance(name, bytes):
                    name = name.decode()
                signatures[name] = list(tensor.shapeSignature)
    return signatures


def _run_tflite(model_path, input_values):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    for detail, value in zip(interpreter.get_input_details(), input_values):
        interpreter.set_tensor(detail["index"], value)
    interpreter.invoke()
    return [interpreter.get_tensor(detail["index"]) for detail in interpreter.get_output_details()]


def _write_tflite(path, batch_size):
    input_layer = tf.keras.layers.Input(shape=(4,), batch_size=batch_size)
    output_layer = tf.keras.layers.Dense(2)(input_layer)
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.Model(input_layer, output_layer))
    path.write_bytes(converter.convert())
    return path


@pytest.fixture(scope="module")
def dynamic_model(tmp_path_factory):
    # batch_size=None keeps the batch dimension dynamic: the converter marks
    # such tensors with a shape signature containing -1.
    return _write_tflite(tmp_path_factory.mktemp("models") / "dynamic.tflite", batch_size=None)


def test_dynamic_fixture_has_shape_signatures(dynamic_model):
    signatures = _shape_signatures(dynamic_model)
    assert signatures
    assert all(-1 in dims for dims in signatures.values())


def test_convert_model_removes_shape_signatures(dynamic_model, tmp_path):
    output = tmp_path / "static.tflite"
    convert_model(dynamic_model, output)

    assert output.is_file()
    assert _shape_signatures(output) == {}


def test_convert_model_preserves_content(dynamic_model, tmp_path):
    output = tmp_path / "static.tflite"
    convert_model(dynamic_model, output)

    before, after = _model_t(dynamic_model), _model_t(output)
    assert len(before.subgraphs) == len(after.subgraphs) == 1
    before_sg, after_sg = before.subgraphs[0], after.subgraphs[0]
    assert list(before_sg.inputs) == list(after_sg.inputs)
    assert list(before_sg.outputs) == list(after_sg.outputs)
    assert len(before_sg.operators) == len(after_sg.operators)
    assert [list(t.shape) for t in before_sg.tensors] == [list(t.shape) for t in after_sg.tensors]
    assert [t.type for t in before_sg.tensors] == [t.type for t in after_sg.tensors]
    assert [t.buffer for t in before_sg.tensors] == [t.buffer for t in after_sg.tensors]
    assert [list(b.data) if b.data is not None else None for b in before.buffers] == [
        list(b.data) if b.data is not None else None for b in after.buffers
    ]


def test_converted_model_runs_at_default_shape(dynamic_model, tmp_path):
    output = tmp_path / "static.tflite"
    convert_model(dynamic_model, output)

    feed = np.ones((1, 4), dtype=np.float32)
    expected, actual = _run_tflite(dynamic_model, [feed]), _run_tflite(output, [feed])
    assert len(expected) == len(actual) == 1
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)


def test_convert_model_noop_on_already_static_model(tmp_path):
    static_in = _write_tflite(tmp_path / "static.tflite", batch_size=1)
    assert _shape_signatures(static_in) == {}

    output = tmp_path / "static-out.tflite"
    convert_model(static_in, output)

    assert _shape_signatures(output) == {}
    interpreter = tf.lite.Interpreter(model_path=str(output))
    interpreter.allocate_tensors()


def test_main_tflite_subcommand(dynamic_model, tmp_path):
    output = tmp_path / "via-main.tflite"
    assert main(["tflite", "-i", str(dynamic_model), "-o", str(output)]) == 0
    assert _shape_signatures(output) == {}


def test_convert_model_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_model(tmp_path / "absent.tflite", tmp_path / "out.tflite")


def test_convert_model_rejects_non_tflite_extension(dynamic_model, tmp_path):
    renamed = tmp_path / "model.bin"
    renamed.write_bytes(dynamic_model.read_bytes())
    with pytest.raises(ValueError):
        convert_model(renamed, tmp_path / "out.tflite")


def test_main_missing_input_fails_cleanly(tmp_path, capsys):
    rc = main(["tflite", "-i", str(tmp_path / "absent.tflite"), "-o", str(tmp_path / "out.tflite")])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("Error:") and "Model file not found" in err
    assert "Traceback" not in err


def test_main_rejects_non_tflite_extension_cleanly(dynamic_model, tmp_path, capsys):
    renamed = tmp_path / "model.bin"
    renamed.write_bytes(dynamic_model.read_bytes())
    rc = main(["tflite", "-i", str(renamed), "-o", str(tmp_path / "out.tflite")])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("Error:") and "Expected a .tflite file" in err
    assert "Traceback" not in err


def test_main_corrupt_model_fails_cleanly(tmp_path, capsys):
    corrupt = tmp_path / "corrupt.tflite"
    corrupt.write_bytes(b"not a tflite model")
    rc = main(["tflite", "-i", str(corrupt), "-o", str(tmp_path / "out.tflite")])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("Error:")
    assert "Traceback" not in err
