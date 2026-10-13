import re
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.testing.onnx import convert_onnx_to_mlir

# Must-fire checks for narrowly matched linalg rewrites. A numerics test still
# passes when a rewrite silently stops matching its target block (e.g. after an
# importer change); these tests read the IR and fail instead. Each check that
# asserts an op is gone also runs a control (the rewrite disabled, or the IR
# before the pass) that asserts the op is there, so a change upstream that
# stops producing the op does not make the check pass by accident.

ONNX_MODELS = Path(__file__).parent / "testdata" / "onnx_models"


def dump_after(torq_compiler, tmp_path, onnx_model, pass_arg, *flags):
    """Compile `onnx_model` and return all IR dumps after `pass_arg`."""
    onnx_path = tmp_path / "model.onnx"
    onnx.save(onnx_model, onnx_path)
    mlir = tmp_path / "model.mlir"
    convert_onnx_to_mlir(str(onnx_path), str(mlir))
    cmd = [
        str(torq_compiler.file_path),
        str(mlir),
        "-o",
        str(tmp_path / "model.vmfb"),
        f"--mlir-print-ir-after={pass_arg}",
        "--torq-hw=SL2610",
        "--mlir-disable-threading",
        *flags,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"
    return proc.stderr


def ir_dumps(text, when):
    """The IR dumps in `text` printed `when` ("Before" or "After") a pass, in order."""
    parts = re.split(r"// -----// IR Dump (Before|After) [^\n]*", text)
    return [body for kind, body in zip(parts[1::2], parts[2::2]) if kind == when]


def gqa_key_scores(heads=4, seq=8, head_dim=16, output_key_t=False):
    """q @ (scale * Expand(key)^T): the grouped-query attention score block
    whose key head broadcast the batch matmul pattern absorbs through the K^T
    transpose and scale. seq != head_dim, so a swapped K^T would not
    type-check. `output_key_t` also returns the transposed key, forking the
    chain."""
    def vi(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    nodes = [
        helper.make_node("Expand", ["key", "heads_shape"], ["key_exp"]),
        helper.make_node("Transpose", ["key_exp"], ["key_t"], perm=[0, 1, 3, 2]),
        helper.make_node("Mul", ["key_t", "scale"], ["key_t_scaled"]),
        helper.make_node("MatMul", ["q", "key_t_scaled"], ["scores"]),
    ]
    graph = helper.make_graph(
        nodes, "gqa_key_scores",
        [vi("q", [1, heads, 1, head_dim]), vi("key", [1, 1, seq, head_dim])],
        [vi("scores", [1, heads, 1, seq])] + ([vi("key_t", [1, heads, head_dim, seq])] if output_key_t else []),
        [numpy_helper.from_array(np.array(head_dim ** -0.5, np.float32), "scale"),
         numpy_helper.from_array(np.array([1, heads, seq, head_dim], np.int64), "heads_shape")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


@pytest.mark.ci
def test_broadcast_matmul_fires_through_key_chain(torq_compiler, tmp_path):
    on = dump_after(torq_compiler, tmp_path, gqa_key_scores(), "torq-optimize-linalg-for-torq")
    # The transpose and scale run on the single key head and the matmul
    # broadcasts it implicitly.
    assert "torq-broadcast-matmul" in on, f"the key head broadcast was not absorbed:\n{on}"
    assert "tensor<1x4x16x8xf32>" not in on, f"the full-size K^T is still materialized:\n{on}"


@pytest.mark.ci
def test_broadcast_matmul_skips_forked_key_chain(torq_compiler, tmp_path):
    # Near miss: the transposed key is also a graph output, so the full-size
    # chain must stay and the broadcast is not absorbed.
    on = dump_after(torq_compiler, tmp_path, gqa_key_scores(output_key_t=True),
                    "torq-optimize-linalg-for-torq")
    assert "torq-broadcast-matmul" not in on, f"the broadcast was absorbed through a forked chain:\n{on}"


def unit_dim_transpose():
    """Transpose [1,64,1,1] -> [1,1,1,64]: only size-1 dims move, so the flat
    element order is unchanged and the transpose is a free reshape."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, 1, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 64])
    node = helper.make_node("Transpose", ["x"], ["y"], perm=[0, 2, 3, 1])
    graph = helper.make_graph([node], "unit_dim_transpose", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.ci
def test_eliminate_transpose_fires(torq_compiler, tmp_path):
    pass_arg = "torq-optimize-linalg-for-torq"
    off = dump_after(torq_compiler, tmp_path, unit_dim_transpose(), pass_arg,
                     "--torq-eliminate-transpose-mode=disable")
    assert "linalg.transpose" in off, f"control: no transpose to eliminate:\n{off}"
    on = dump_after(torq_compiler, tmp_path, unit_dim_transpose(), pass_arg)
    assert "linalg.transpose" not in on, f"the data-preserving transpose was kept:\n{on}"


@pytest.mark.ci
def test_collapse_unrolled_concat_fires(torq_compiler, tmp_path):
    # 40 Slice -> Squeeze -> Unsqueeze chains of v[1,40,8] plus one extra operand,
    # concatenated: the 40 unit slices cover v, so the concat reads v directly.
    model = onnx.load(ONNX_MODELS / "unrolled_concat.onnx")
    pass_arg = "torq-collapse-unrolled-concat"
    off = dump_after(torq_compiler, tmp_path, model, pass_arg,
                     "--torq-collapse-unrolled-concat-mode=disable")
    assert off.count("tensor.extract_slice") >= 40, f"control: no unrolled stack:\n{off}"
    on = dump_after(torq_compiler, tmp_path, model, pass_arg)
    assert "tensor.extract_slice" not in on, f"the unit slices were not collapsed:\n{on}"
    assert "tensor<1x40x8xf32>) -> tensor<1x41x8xf32>" in on, \
        f"the concat does not read the source tensor:\n{on}"


@pytest.mark.ci
def test_singleton_gather_unsqueeze_restores_rank(torq_compiler, tmp_path):
    # Gather(axis=1, 0) -> Sigmoid -> Unsqueeze(axes=[1]) over [1,1,256]: the
    # importer runs the sigmoid at [1,256] between a collapse and an expand. The
    # pre-dispatch run of the pass (the first one) folds the round trip, so the
    # sigmoid runs at [1,1,256].
    model = onnx.load(ONNX_MODELS / "singleton_gather_unsqueeze.onnx")
    pass_arg = "torq-fold-unit-extent-wrapper"
    text = dump_after(torq_compiler, tmp_path, model, pass_arg,
                      f"--mlir-print-ir-before={pass_arg}")
    before, after = ir_dumps(text, "Before")[0], ir_dumps(text, "After")[0]
    assert "tensor.collapse_shape" in before and "tensor.expand_shape" in before, \
        f"control: no unit-dim round trip to fold:\n{before}"
    assert "tensor.collapse_shape" not in after and "tensor.expand_shape" not in after, \
        f"the unit-dim round trip was not folded:\n{after}"
    assert "linalg.generic" in after and "-> tensor<1x1x256xf32>" in after, \
        f"the sigmoid does not run at the rank of its surrounding tensors:\n{after}"
