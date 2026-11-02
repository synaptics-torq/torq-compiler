import subprocess

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.testing.iree import _find_iree_tool
from torq.testing.onnx import convert_onnx_to_mlir

"""Checks that the torq input pipeline expands onnx.GRU with the Torq gate split.

torch-mlir expands onnx.GRU in convert-torch-onnx-to-torch. Its tests check the
math. These tests check that torq sets gru-split-gates-min-elements to
kMinElementsForSlicing (2000), and that reverse and bidirectional GRUs lower.
"""

# name, seq_len, batch, hidden_size, direction, aten.linear count, aten.sigmoid count
CASES = [
    # batch 1: the gate slices are contiguous, so the matmuls stay packed
    ("batch1_reverse", 2, 1, 16, "reverse", 2, 1),
    # T*B*3H = 384 and B*3H = 96 are below 2000: packed, one loop per direction
    ("small_bidirectional", 4, 2, 16, "bidirectional", 4, 2),
    # T*B*3H = 6144 and B*3H = 3072 reach 2000: one matmul for each gate
    ("large_batch_reverse", 2, 64, 16, "reverse", 6, 2),
]


def _make_gru(path, seq_len, batch, input_size, hidden, direction):
    rng = np.random.default_rng(0)
    dirs = 2 if direction == "bidirectional" else 1
    inits = [
        numpy_helper.from_array(rng.standard_normal((dirs, 3 * hidden, input_size)).astype(np.float32), "W"),
        numpy_helper.from_array(rng.standard_normal((dirs, 3 * hidden, hidden)).astype(np.float32), "R"),
        numpy_helper.from_array(rng.standard_normal((dirs, 6 * hidden)).astype(np.float32), "B"),
    ]
    node = helper.make_node("GRU", ["X", "W", "R", "B"], ["Y", "Y_h"], hidden_size=hidden,
                            linear_before_reset=1, direction=direction)
    graph = helper.make_graph(
        [node], "gru",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [seq_len, batch, input_size])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [seq_len, dirs, batch, hidden]),
         helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [dirs, batch, hidden])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.mark.ci
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_onnx_gru_expansion_in_torq_input_pipeline(tmp_path, case):
    _, seq_len, batch, hidden, direction, linears, sigmoids = case
    onnx_path = tmp_path / "gru.onnx"
    _make_gru(onnx_path, seq_len, batch, 8, hidden, direction)
    mlir = tmp_path / "gru.mlir"
    convert_onnx_to_mlir(onnx_path, mlir)

    result = subprocess.run(
        [_find_iree_tool("TORQ_COMPILE", "torq-compile"), str(mlir), "--compile-to=input",
         "-o", str(tmp_path / "input.mlir"),
         "--mlir-print-ir-after=convert-torch-onnx-to-torch"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # torch-mlir keeps the timestep loop, so the loop body appears once per direction
    converted = result.stderr
    assert '"onnx.GRU"' not in converted
    assert converted.count("torch.aten.linear") == linears
    assert converted.count("torch.aten.sigmoid") == sigmoids
