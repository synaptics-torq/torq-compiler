import re
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.testing.onnx import convert_onnx_to_mlir

try:
    from iree.compiler import ir as _iree_ir
except ImportError:
    _iree_ir = None

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


# The rewrites below expect the bf16 IR the dtype conversion produces.
CONVERT_DTYPES = ("--torq-convert-dtypes", "--torq-convert-io-dtype")


def matmul_model(name, a_shape, b_shape, out_shape, dtype=TensorProto.FLOAT):
    """A @ B with static shapes, both operands graph inputs."""
    vi = lambda n, s: helper.make_tensor_value_info(n, dtype, s)
    node = helper.make_node("MatMul", ["a", "b"], ["o"], name="m")
    graph = helper.make_graph(
        [node], name, [vi("a", a_shape), vi("b", b_shape)], [vi("o", out_shape)])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


# The named matmul ops the pattern matches on, and the elementwise-mul broadcast
# map it emits (rank-3 drops the row dim of the RHS, rank-2 drops the leading dim).
NAMED_MATMUL = re.compile(r"linalg\.(batch_)?matmul ")
BROADCAST_MAP_R3 = "affine_map<(d0, d1, d2) -> (d0, 0, d2)>"
BROADCAST_MAP_R2 = "affine_map<(d0, d1) -> (0, d1)>"


def assert_fired(dump, map_substr=None):
    assert not NAMED_MATMUL.search(dump), \
        f"FoldScalarMatmulPattern did not remove the unit-reduction matmul:\n{dump}"
    assert "arith.mulf" in dump, f"the elementwise product is missing:\n{dump}"
    if map_substr:
        assert map_substr in dump, f"the expected generic form is missing:\n{dump}"


# With B > 1 the pass's elementwise binary broadcast normalization follows the
# rewrite: it materializes the multi-element RHS broadcast and levels the
# product's maps (it keeps the map form only for the single-element B == 1
# case). Both end forms are the elementwise product; assert the marker.
BROADCASTED = "broadcasted = true"


def assert_not_fired(dump, matmul_substr, map_substr):
    assert NAMED_MATMUL.search(dump) and matmul_substr in dump, \
        f"the matmul must survive the pass but is gone:\n{dump}"
    assert map_substr not in dump, \
        f"FoldScalarMatmulPattern fired on a near miss:\n{dump}"


@pytest.mark.ci
def test_fold_scalar_matmul_fires_rank3(torq_compiler, tmp_path):
    # A [1,16,1] @ B [1,1,1]: both inputs, the RoPE site shape.
    model = matmul_model("fold_r3", [1, 16, 1], [1, 1, 1], [1, 16, 1])
    on = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq", *CONVERT_DTYPES)
    assert_fired(on, BROADCAST_MAP_R3)
    # The explicit enable mode is the default behavior.
    on = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq",
                    *CONVERT_DTYPES, "--torq-fold-scalar-matmul-mode=enable")
    assert_fired(on, BROADCAST_MAP_R3)
    # Disabled: the pre-rewrite named matmul survives.
    off = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq",
                     *CONVERT_DTYPES, "--torq-fold-scalar-matmul-mode=disable")
    assert_not_fired(off, "tensor<1x16x1xbf16>, tensor<1x1x1xbf16>", BROADCAST_MAP_R3)


@pytest.mark.ci
def test_fold_scalar_matmul_fires_rank2(torq_compiler, tmp_path):
    model = matmul_model("fold_r2", [9, 1], [1, 1], [9, 1])
    on = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq", *CONVERT_DTYPES)
    assert_fired(on, BROADCAST_MAP_R2)
    off = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq",
                     *CONVERT_DTYPES, "--torq-fold-scalar-matmul-mode=disable")
    assert_not_fired(off, "tensor<9x1xbf16>, tensor<1x1xbf16>", BROADCAST_MAP_R2)


@pytest.mark.ci
def test_fold_scalar_matmul_fires_rank3_batch_gt_1(torq_compiler, tmp_path):
    # Generalization G1: batch extent B > 1 (observed B = 1 at the target site).
    model = matmul_model("fold_r3_b", [3, 4, 1], [3, 1, 1], [3, 4, 1])
    on = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq", *CONVERT_DTYPES)
    assert_fired(on, BROADCASTED)


@pytest.mark.ci
def test_fold_scalar_matmul_fires_on_rank4_onnx(torq_compiler, tmp_path):
    # A rank-4 ONNX matmul with a unit leading batch dim is lowered to a named
    # rank-3 batch_matmul by the importer, so it is a G1 positive (any batch
    # extent), not a distinct op kind: the rewrite must fire.
    model = matmul_model("fold_r4", [1, 4, 11, 1], [1, 4, 1, 1], [1, 4, 11, 1])
    on = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq", *CONVERT_DTYPES)
    assert_fired(on, BROADCASTED)


@pytest.mark.ci
def test_fold_scalar_matmul_skips_row_scale(torq_compiler, tmp_path):
    # Near miss N1: k == 1 but n > 1 — the unit reduction sits on the LHS of a
    # genuine row-scale outer product; both ranks must not fire.
    for name, a, b, o, matmul_substr, map_substr in [
        ("n1_r2", [9, 1], [1, 4], [9, 4], "tensor<9x1xbf16>, tensor<1x4xbf16>", BROADCAST_MAP_R2),
        ("n1_r3", [1, 9, 1], [1, 1, 4], [1, 9, 4], "tensor<1x9x1xbf16>, tensor<1x1x4xbf16>", BROADCAST_MAP_R3),
    ]:
        on = dump_after(torq_compiler, tmp_path, matmul_model(name, a, b, o),
                        "torq-optimize-linalg-for-torq", *CONVERT_DTYPES)
        assert_not_fired(on, matmul_substr, map_substr)


@pytest.mark.ci
def test_fold_scalar_matmul_skips_matvec(torq_compiler, tmp_path):
    # Near miss N2: n == 1 but k > 1 — a matrix-vector contraction; both ranks
    # must not fire.
    for name, a, b, o, matmul_substr, map_substr in [
        ("n2_r2", [9, 4], [4, 1], [9, 1], "tensor<9x4xbf16>, tensor<4x1xbf16>", BROADCAST_MAP_R2),
        ("n2_r3", [1, 9, 4], [1, 4, 1], [1, 9, 1], "tensor<1x9x4xbf16>, tensor<1x4x1xbf16>", BROADCAST_MAP_R3),
    ]:
        on = dump_after(torq_compiler, tmp_path, matmul_model(name, a, b, o),
                        "torq-optimize-linalg-for-torq", *CONVERT_DTYPES)
        assert_not_fired(on, matmul_substr, map_substr)


# Dispatch-shaped functions at the pass's per-dispatch point: the named
# unit-reduction matmuls (rank-3 and rank-2) over their zero-fill inits, exactly
# as the pipeline emits them. Drives OptimizeLinalgForTorq standalone (via
# iree-opt) to prove the pass-scoped label: disable-patterns=
# "FoldScalarMatmulPattern" must disable both sub-patterns with the mode flag
# left at its default (enable).
FOLD_LABEL_MODULE = """
func.func @fold_label_r3() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(4096) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x1xbf16>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(4096) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x1xbf16>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(4096) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x16x1xbf16>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 16, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x1xbf16>> -> tensor<1x16x1xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 1, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x1xbf16>> -> tensor<1x1x1xbf16>
    %5 = tensor.empty() : tensor<1x16x1xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<1x16x1xbf16>) -> tensor<1x16x1xbf16>
    %7 = linalg.batch_matmul ins(%3, %4 : tensor<1x16x1xbf16>, tensor<1x1x1xbf16>) outs(%6 : tensor<1x16x1xbf16>) -> tensor<1x16x1xbf16>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [1, 16, 1], strides = [1, 1, 1] : tensor<1x16x1xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x16x1xbf16>>
    return
}
func.func @fold_label_r2() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(4096) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x1xbf16>>
    %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(4096) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1xbf16>>
    %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(4096) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<9x1xbf16>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [9, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x1xbf16>> -> tensor<9x1xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1xbf16>> -> tensor<1x1xbf16>
    %5 = tensor.empty() : tensor<9x1xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<9x1xbf16>) -> tensor<9x1xbf16>
    %7 = linalg.matmul ins(%3, %4 : tensor<9x1xbf16>, tensor<1x1xbf16>) outs(%6 : tensor<9x1xbf16>) -> tensor<9x1xbf16>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [9, 1], strides = [1, 1] : tensor<9x1xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<9x1xbf16>>
    return
}
"""


def run_optimize_pass_standalone(torq_compiler, tmp_path, module_text, pass_spec):
    """Run OptimizeLinalgForTorq via iree-opt on module text; return the IR."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    mlir = tmp_path / "module.mlir"
    mlir.write_text(module_text)
    cmd = [str(iree_opt), str(mlir),
           "--pass-pipeline", f"builtin.module(func.func({pass_spec}))"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"iree-opt failed:\n{proc.stderr}"
    return proc.stdout


@pytest.mark.ci
def test_fold_scalar_matmul_disable_patterns_label(torq_compiler, tmp_path):
    # Default mode (enable): both sub-patterns fire in the standalone pass.
    on = run_optimize_pass_standalone(
        torq_compiler, tmp_path, FOLD_LABEL_MODULE, "torq-optimize-linalg-for-torq")
    assert not NAMED_MATMUL.search(on), \
        f"the standalone pass with default mode must fold both matmuls:\n{on}"
    # The pass-scoped label disables both sub-patterns at once (the shared
    # setDebugName), with the mode flag left at its default.
    off = run_optimize_pass_standalone(
        torq_compiler, tmp_path, FOLD_LABEL_MODULE,
        'torq-optimize-linalg-for-torq{disable-patterns="FoldScalarMatmulPattern"}')
    r3 = off.count("linalg.batch_matmul ")
    r2 = off.count("linalg.matmul ")
    assert r3 == 1 and r2 == 1, \
        f"the pass-scoped label must leave both named matmuls intact (got {r3} rank-3, {r2} rank-2):\n{off}"


# --- Per-tensor dequantized constant weight --------------------------------
#
# BatchMatmulOpPattern rewrites a batch matmul input lifted from a per-tensor
# DequantizeLinear of a constant weight into the single-block form the
# block-quantized matmul lowerings consume, so the weight stays int8 and the
# decode matmul becomes a mixed-quant matmul.


def dequant_model(name, signed=False):
    """MatMul(h [1,1,8], DequantizeLinear(W [8,16], scale, zp)) -> [1,1,16].
    The unsigned weight has 128 distinct values incl. 0, the zero point (129),
    and 255; signed=True uses the same bytes as int8 with zero point zero."""
    w = np.concatenate([[0, 128, 129, 130, 254, 255], np.arange(122, 0, -1)]).astype(np.uint8).reshape(8, 16)
    scale, zp = np.float32(0.011408232152462006), np.uint8(129)
    if signed:
        w, zp = w.view(np.int8), np.int8(0)
    nodes = [helper.make_node("DequantizeLinear", ["W", "scale", "zp"], ["deq"]),
             helper.make_node("MatMul", ["h", "deq"], ["y"], name="m")]
    inits = [numpy_helper.from_array(w, "W"), numpy_helper.from_array(scale, "scale"),
             numpy_helper.from_array(zp, "zp")]
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, 1, 8])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 16])], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


@pytest.mark.ci
@pytest.mark.parametrize("signed", [False, True])
def test_per_tensor_dequant_matmul_lowers_to_mixed_quant(torq_compiler, tmp_path, signed):
    dump = dump_after(torq_compiler, tmp_path, dequant_model("dequant_proj", signed=signed),
                      "torq-linalg-to-torqhl-conversion", *CONVERT_DTYPES)
    assert "torq_hl.mixed_quant_matmul" in dump, f"the matmul did not become a mixed-quant matmul:\n{dump}"
    assert "tensor<1x8x16xi8>" in dump, f"the int8 weight is not the single-block [1,K,N] form:\n{dump}"


# The per-dispatch form of the real projection: the weight lifted by
# expand_shape, scale and zero point captured as scalar constants. The weight
# bytes are 0, 128, 129, 130, 254, 255 and 6 more.
DEQUANT_MODULE = """
func.func @dequant_matmul(%h: tensor<1x1x4xbf16>) -> tensor<1x1x3xbf16> {
  %w = arith.constant dense<"0x00808182FEFF05060708090A"> : tensor<4x3xi8>
  %scale = arith.constant 1.140823e-02 : bf16
  %zp = arith.constant 129 : i32
  %zero = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<4x3xbf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%w : tensor<4x3xi8>) outs(%0 : tensor<4x3xbf16>) {
  ^bb0(%in: i8, %out: bf16):
    %3 = arith.extui %in : i8 to i32
    %4 = arith.subi %3, %zp : i32
    %5 = arith.sitofp %4 : i32 to bf16
    %6 = arith.mulf %5, %scale : bf16
    linalg.yield %6 : bf16
  } -> tensor<4x3xbf16>
  %2 = tensor.expand_shape %1 [[0, 1], [2]] output_shape [1, 4, 3] : tensor<4x3xbf16> into tensor<1x4x3xbf16>
  %7 = tensor.empty() : tensor<1x1x3xbf16>
  %8 = linalg.fill ins(%zero : bf16) outs(%7 : tensor<1x1x3xbf16>) -> tensor<1x1x3xbf16>
  %9 = linalg.batch_matmul ins(%h, %2 : tensor<1x1x4xbf16>, tensor<1x4x3xbf16>) outs(%8 : tensor<1x1x3xbf16>) -> tensor<1x1x3xbf16>
  return %9 : tensor<1x1x3xbf16>
}
"""


@pytest.mark.ci
def test_per_tensor_dequant_matmul_single_block_form(torq_compiler, tmp_path):
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, DEQUANT_MODULE, "torq-optimize-linalg-for-torq")
    # uint8 w - 129 == int8 (w ^ 0x80) - 1: the weight is re-encoded as int8
    # with zero point 1, and scale/zero point become [1,1,N] splats.
    assert ("dense<[[[-128, 0, 1], [2, 126, 127], [-123, -122, -121], [-120, -119, -118]]]> "
            ": tensor<1x4x3xi8>") in out, out
    assert "dense<1> : tensor<1x1x3xi8>" in out and "tensor<1x1x3xbf16>" in out, out
    assert "arith.extui" not in out and "arith.extsi" in out, out


@pytest.mark.ci
@pytest.mark.parametrize("variant", ["runtime_weight", "zero_point_256", "zero_point_negative"])
def test_per_tensor_dequant_matmul_near_misses(torq_compiler, tmp_path, variant):
    module = DEQUANT_MODULE
    if variant == "runtime_weight":
        module = module.replace("(%h: tensor<1x1x4xbf16>)", "(%h: tensor<1x1x4xbf16>, %rw: tensor<4x3xi8>)")
        module = module.replace("ins(%w : tensor<4x3xi8>)", "ins(%rw : tensor<4x3xi8>)")
    else:
        module = module.replace("129 : i32", "256 : i32" if variant == "zero_point_256" else "-1 : i32")
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, module, "torq-optimize-linalg-for-torq")
    assert "arith.extui" in out and "xi8>" in out and "1x4x3xi8" not in out, out


# --- Float ACT tail fold (LinalgToTorqHL conversion) -------------------------
#
# The checks below pin down foldForwardScaleClamp's float tail walk (a
# [f32->bf16 truncf] [scalar const add] [clamp(s)] chain absorbed into the
# consumer's ACT) and its consumers. Unlike the ONNX-driven checks above, the
# input is the exact linalg chain the patterns see, so no importer sits in
# between; each testdata file's header states the expected end IR.

LINALG_TO_TORQHL = "torq-linalg-to-torqhl-conversion"
PATTERN_FIRES_DIR = Path(__file__).parent / "testdata" / "linalg_pattern_fires"
HARDSIGMOID = Path(__file__).parent / "testdata" / "linalg_ops" / "hardsigmoid-bf16.mlir"


def count_ops(ir_text, op):
    # torq_hl ops print in generic form ("torq_hl.mul"); linalg.generic in
    # pretty form.
    if op.startswith("torq_hl."):
        return len(re.findall(rf'"{re.escape(op)}"', ir_text))
    return len(re.findall(rf"\b{re.escape(op)}\b", ir_text))


def op_lines(ir_text, op):
    if op.startswith("torq_hl."):
        pat = rf'"{re.escape(op)}"'
    else:
        pat = rf"\b{re.escape(op)}\b"
    return [line for line in ir_text.splitlines() if re.search(pat, line)]


@pytest.mark.ci
def test_hardsigmoid_chain_folds_to_single_mul(torq_compiler, tmp_path):
    """The four-generic HardSigmoid chain (mul -> add -> min -> max) must
    become one torq_hl.mul whose ACT carries the beta bias and the [0, 1]
    clip, with no linalg left."""
    out = run_optimize_pass_standalone(torq_compiler, tmp_path,
                                       HARDSIGMOID.read_text(), LINALG_TO_TORQHL)
    assert count_ops(out, "torq_hl.mul") == 1, out
    assert count_ops(out, "linalg.generic") == 0, out
    mul = op_lines(out, "torq_hl.mul")[0]
    # clip: min 0.0f, max 1.0f (f32 bit patterns)
    assert "output_min = 0 : i32" in mul, mul
    assert "output_max = 1065353216 : i32" in mul, mul
    # the beta bias is recorded in f32
    assert "dense<5.000000e-01> : tensor<1xf32>" in out, out


@pytest.mark.ci
def test_perchannel_mul_bias_absorbs_scalar_add_and_clamp(torq_compiler, tmp_path):
    """per-channel mulf + addf (same fuse group) + scalar add + ReLU must
    become one torq_hl.mul: the scalar add is absorbed into the per-channel
    bias constant (rebuilt in f32) and the ReLU becomes the clip.

    Catches both the drop of the tail's scalar add (the rebuilt bias would
    read [0.1, 0.2, 0.3, 0.4] instead of [0.6, 0.7, 0.8, 0.9]) and the
    dangling per-channel add (the conversion crashes).
    """
    case = (PATTERN_FIRES_DIR / "perchannel-mul-bias-scalar-add-clamp-bf16.mlir").read_text()
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, case, LINALG_TO_TORQHL)
    assert count_ops(out, "torq_hl.mul") == 1, out
    assert count_ops(out, "torq_hl.add") == 0, out
    assert count_ops(out, "linalg.generic") == 0, out
    # bias rebuilt in f32 with the scalar add (0.5) absorbed
    assert "dense<[0.600097656, 0.700195313, 0.80078125, 0.900390625]> : tensor<4xf32>" in out, out
    mul = op_lines(out, "torq_hl.mul")[0]
    assert "output_min = 0 : i32" in mul, mul
    # ReLU: open top at the largest finite f32
    assert "output_max = 2139095039 : i32" in mul, mul


@pytest.mark.ci
def test_perchannel_mul_resblob_bias_keeps_scalar_add_tail(torq_compiler, tmp_path):
    """When the per-channel bias is a dense_resource blob (how ONNX
    initializers arrive), it cannot be rebuilt to absorb the tail's scalar
    add: the tail must be kept alive (a torq_hl.add with the +0.5 bias and
    the ReLU clip), not silently dropped from the single torq_hl.mul.
    """
    if _iree_ir is None:
        pytest.skip("iree python bindings unavailable")
    text = (PATTERN_FIRES_DIR / "perchannel-mul-bias-scalar-add-clamp-bf16.mlir").read_text()
    placeholder = "%bias = arith.constant dense<0.000000e+00> : tensor<4xbf16>"
    text = text.replace(
        "%bias = arith.constant dense<[0.100000e+00, 0.200000e+00, 0.300000e+00, 0.400000e+00]> : tensor<4xbf16>",
        placeholder,
    )
    assert placeholder in text
    # dense_resource<perch_bias> : tensor<4xbf16>, values [1.0, 2.0, 3.0, 4.0]
    with _iree_ir.Context() as ctx, _iree_ir.Location.unknown(ctx) as loc:
        ty = _iree_ir.RankedTensorType.get([4], _iree_ir.BF16Type.get())
        attr = _iree_ir.DenseResourceElementsAttr.get_from_buffer(
            ml_dtypes.bfloat16(np.array([1, 2, 3, 4], dtype=np.int16)).view(np.uint16).tobytes(),
            "perch_bias", ty,
        )
        mod = _iree_ir.Module.parse(text)
        block = mod.body.operations[0].regions[0].blocks[0]
        for op in block.operations:
            if op.name != "arith.constant":
                continue
            cst = op.attributes["value"]
            if cst.type == ty and str(cst) == "dense<0.000000e+00> : tensor<4xbf16>":
                op.attributes["value"] = attr
                break
        else:
            raise AssertionError("bias placeholder constant not found")
        mlir_file = tmp_path / "perchannel-resblob.mlir"
        with open(mlir_file, "wb") as fh:
            mod.operation.write_bytecode(fh)
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [str(iree_opt), str(mlir_file),
         "--pass-pipeline", f"builtin.module(func.func({LINALG_TO_TORQHL}))"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"conversion failed:\n{proc.stderr}"
    out = proc.stdout
    # the blob bias survives (not rebuilt) behind an f2f act feeding the mul
    assert "dense_resource<perch_bias>" in out, out
    assert count_ops(out, "torq_hl.mul") == 1, out
    assert count_ops(out, "torq_hl.add") == 1, out
    assert count_ops(out, "linalg.generic") == 0, out
    # the kept tail: the scalar add and the ReLU live on the add
    add = op_lines(out, "torq_hl.add")[0]
    assert "output_min = 0 : i32" in add, add
    assert "output_max = 2139095039 : i32" in add, add
    assert "dense<5.000000e-01> : tensor<1xf32>" in out, out


@pytest.mark.ci
def test_conv1d_folds_truncf_add_clamp_tail(torq_compiler, tmp_path):
    """A Conv1D (5D generic) followed by truncf + scalar add + ReLU must
    become one torq_hl.conv1d whose ACT carries the +0.5 bias (per channel,
    f32) and the [0, .] clip, with a bf16 output feeding the Kw reduce.

    The conv1d pattern runs in the pre-conversion, whose bias probe evaluates
    constants through the full compile context (standalone iree-opt cannot
    load the dialects it needs), so this goes through torq-compile (with
    slicing disabled, which would pull the tail out of the dispatch).
    """
    case = PATTERN_FIRES_DIR / "conv1d-add-clamp-tail-bf16.mlir"
    cmd = [str(torq_compiler.file_path), str(case), "-o", str(tmp_path / "model.vmfb"),
           "--torq-disable-host", "--torq-disable-css", "--torq-hw=SL2610",
           "--torq-disable-slicing", "--mlir-disable-threading",
           f"--mlir-print-ir-after=torq-linalg-to-torqhl-pre-conversion"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    dumps = [d for d in ir_dumps(proc.stderr, "After") if "main_dispatch" in d]
    assert dumps, "no dump of the dispatch after torq-linalg-to-torqhl-pre-conversion"
    dump = dumps[-1]
    assert count_ops(dump, "torq_hl.conv1d") == 1, dump
    assert count_ops(dump, "linalg.generic") == 0, dump
    conv = op_lines(dump, "torq_hl.conv1d")[0]
    # bf16 output (the truncf tail was absorbed) with the ReLU clip
    assert "-> tensor<1x2x1x3x3xbf16>" in conv, conv
    assert "output_min = 0 : i32" in conv, conv
    assert "output_max = 2139095039 : i32" in conv, conv
    # the scalar add folded into the f32 per-channel bias (F=2)
    assert "dense<5.000000e-01> : tensor<2xf32>" in dump, dump


@pytest.mark.ci
def test_tail_stops_at_width_converting_clamp(torq_compiler, tmp_path):
    """A clamp that converts width (ReLU in bf16, extf to the f32 result --
    the form the elementwise fusion leaves at a model's f32 boundary) must
    stop the forward float tail fold: the mul keeps its bf16 result, and the
    clamp lowers on its own to a bf16-in / f32-out act.
    """
    case = (PATTERN_FIRES_DIR / "clamp-extf-result-bf16.mlir").read_text()
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, case, LINALG_TO_TORQHL)
    assert count_ops(out, "torq_hl.mul") == 1, out
    assert count_ops(out, "linalg.generic") == 0, out
    mul = op_lines(out, "torq_hl.mul")[0]
    # the tail ended on an f32 type pre-fix, turning this into an
    # f32-result mul (or invalid IR); it must stay bf16
    assert "-> tensor<1x4x1x4xbf16>" in mul, mul
    # the width-converting clamp survived as its own act
    acts = op_lines(out, "torq_hl.act")
    assert any("name = \"clamp\"" in a and "-> tensor<1x4x1x4xf32>" in a for a in acts), out
