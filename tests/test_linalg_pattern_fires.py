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


def float_initializer(name, values, dtype):
    """A float initializer of the ONNX `dtype` (FLOAT or BFLOAT16)."""
    return numpy_helper.from_array(np.asarray(values, helper.tensor_dtype_to_np_dtype(dtype)), name)


def gqa_key_scores(heads=4, seq=8, head_dim=16, output_key_t=False, dtype=TensorProto.FLOAT):
    """q @ (scale * Expand(key)^T): the grouped-query attention score block
    whose key head broadcast the batch matmul pattern absorbs through the K^T
    transpose and scale. seq != head_dim, so a swapped K^T would not
    type-check. `output_key_t` also returns the transposed key, forking the
    chain."""
    def vi(name, shape):
        return helper.make_tensor_value_info(name, dtype, shape)

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
        [float_initializer("scale", head_dim ** -0.5, dtype),
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


# --- Grouped head broadcast (the rank-4 cache source) -----------------------
#
# BatchMatmulOpPattern absorbs a batch matmul operand built by repeating the
# heads of a [1,KV,S,D] key/value cache over a group of G query heads (the
# unedited grouped-query attention form), so the full-size K^T and the
# expanded cache are not materialized: query heads sharing a cache head
# are folded into the matmul's M dimension. Both checks below drive the pass standalone on
# the exact dispatch-shaped chain, so no importer sits in between.

def grouped_k_module(
    kv=2,
    perm=(0, 1, 3, 2),
    m=1,
    g=3,
    fork_broadcast=False,
    body_reads_init=False,
    wrong_merge=False,
):
    """One score matmul: q @ (scale * repeat(k_cache)^T). KV = kv, G = g,
    H = KV*G, S = 7, D = 5 (non-square so a swapped K^T would not type-check).
    `kv = 1` is the single-cache-head (Form B) source shape the producer
    keeps at rank 4. `perm` is the K^T link's permutation; `wrong_merge`
    splits the merge collapse's head reassociation; `fork_broadcast` also
    returns the rank-5 broadcast result; `body_reads_init` turns the scale
    link into an accumulate form."""
    s, d = 7, 5
    h = kv * g
    # The wrong-merge chain collapses to a [KV,.,.] operand, so the matmul's
    # batch extent is KV there (q must match it for the IR to be valid).
    q_batch = kv if wrong_merge else h
    merge_src = f"tensor<1x{kv}x{g}x{s}x{d}xf32>"
    if wrong_merge:
        # [[0],[1],[2,3],[4]]: the head dim is not merged with the group dim,
        # so the link stack runs over [1,KV,G*S,D] and the batch collapse
        # yields a [KV,.,.] operand whose batch is not the head count.
        merge_out = f"tensor<1x{kv}x{g * s}x{d}xf32>"
        merge_reassoc = "[[0], [1], [2, 3], [4]]"
        t_in = f"1x{kv}x{g * s}x{d}"
        t_out = f"1x{kv}x{d}x{g * s}"
        b_shape = f"tensor<{kv}x{d}x{g * s}xf32>"
        q_k, out_n = d, g * s
    else:
        merge_out = f"tensor<1x{h}x{s}x{d}xf32>"
        merge_reassoc = "[[0], [1, 2], [3], [4]]"
        # The link perm applied to the merge result [1,H,S,D]; the batch
        # collapse [[0,1],[2],[3]] then fixes the matmul's B shape and q's
        # reduction dim. (0,1,3,2) is the observed K^T; the other perms are
        # near-miss boundaries (all type-valid).
        in_dims = [1, h, s, d]
        t_out_dims = [in_dims[p] for p in perm]
        t_in = f"1x{h}x{s}x{d}"
        t_out = "x".join(str(x) for x in t_out_dims)
        b_dims = [t_out_dims[0] * t_out_dims[1], t_out_dims[2], t_out_dims[3]]
        b_shape = f"tensor<{b_dims[0]}x{b_dims[1]}x{b_dims[2]}xf32>"
        q_k, out_n = b_dims[1], b_dims[2]
    q_shape = f"tensor<{q_batch}x{m}x{q_k}xf32>"
    out_shape = f"tensor<{q_batch}x{m}x{out_n}xf32>"
    yield_init = (
        "%mul = arith.mulf %in, %in_0 : f32\n    %add = arith.addf %out, %mul : f32\n    linalg.yield %add : f32"
        if body_reads_init
        else "%mul = arith.mulf %in, %in_0 : f32\n    linalg.yield %mul : f32"
    )
    fork_ret = ", %1" if fork_broadcast else ""
    fork_out = f", {merge_src}" if fork_broadcast else ""
    return f"""
func.func @grouped_k(%q: {q_shape}, %k: tensor<1x{kv}x{s}x{d}xf32>) -> ({out_shape}{fork_out}) {{
  %scale = arith.constant dense<0.5> : tensor<1xf32>
  %zero = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : {merge_src}
  %1 = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}} ins(%k : tensor<1x{kv}x{s}x{d}xf32>) outs(%0 : {merge_src}) {{
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }} -> {merge_src}
  %2 = tensor.collapse_shape %1 {merge_reassoc} : {merge_src} into {merge_out}
  %3 = tensor.empty() : tensor<{t_out}xf32>
  %t = linalg.transpose ins(%2 : tensor<{t_in}xf32>) outs(%3 : tensor<{t_out}xf32>) permutation = [{perm[0]}, {perm[1]}, {perm[2]}, {perm[3]}]
  %4 = tensor.empty() : tensor<{t_out}xf32>
  %sc = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%t, %scale : tensor<{t_out}xf32>, tensor<1xf32>) outs(%4 : tensor<{t_out}xf32>) {{
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    {yield_init}
  }} -> tensor<{t_out}xf32>
  %c = tensor.collapse_shape %sc [[0, 1], [2], [3]] : tensor<{t_out}xf32> into {b_shape}
  %f = tensor.empty() : {out_shape}
  %acc = linalg.fill ins(%zero : f32) outs(%f : {out_shape}) -> {out_shape}
  %mm = linalg.batch_matmul ins(%q, %c : {q_shape}, {b_shape}) outs(%acc : {out_shape}) -> {out_shape}
  return %mm{fork_ret} : {out_shape}{fork_out}
}}
"""


@pytest.mark.ci
def test_grouped_head_broadcast_fires(torq_compiler, tmp_path):
    out = run_optimize_pass_standalone(
        torq_compiler, tmp_path, grouped_k_module(), "torq-optimize-linalg-for-torq"
    )
    assert "floordiv" not in out, out
    assert "tensor<2x3x5xf32>" in out, f"heads were not folded into M:\n{out}"
    assert "tensor<1x6x5x7xf32>" not in out, f"full-size K^T remains:\n{out}"
    assert "tensor<2x5x7xf32>" in out, f"K^T must run on the small cache:\n{out}"


@pytest.mark.ci
def test_grouped_head_broadcast_disable_mode(torq_compiler, tmp_path):
    # Control: the fold fires with the default mode and is absent when the
    # pattern's pass-scoped label disables it (the label covers the
    # pre-existing rank-3 form and this one).
    on = run_optimize_pass_standalone(
        torq_compiler, tmp_path, grouped_k_module(), "torq-optimize-linalg-for-torq"
    )
    disabled = run_optimize_pass_standalone(
        torq_compiler, tmp_path, grouped_k_module(),
        'torq-optimize-linalg-for-torq{disable-patterns="BatchMatmulOpPattern"}')
    assert "tensor<2x3x5xf32>" in on
    assert "tensor<1x2x3x7x5xf32>" in disabled, disabled
    assert "linalg.batch_matmul" in disabled, f"the matmul must survive:\n{disabled}"


@pytest.mark.ci
def test_grouped_head_broadcast_formb_fires(torq_compiler, tmp_path):
    # Single-cache-head and grouped-cache forms now use the same rewrite.
    out = run_optimize_pass_standalone(
        torq_compiler, tmp_path, grouped_k_module(kv=1), "torq-optimize-linalg-for-torq"
    )
    assert "tensor<1x3x5xf32>" in out, f"the single-head chain was not folded:\n{out}"
    assert "floordiv" not in out, f"head folding must not introduce floordiv maps:\n{out}"
    assert "tensor<1x1x3x7x5xf32>" not in out, f"the expanded cache is still materialized:\n{out}"
    assert "tensor<1x5x7xf32>" in out, f"the K^T does not run on the squeezed cache:\n{out}"
    off = run_optimize_pass_standalone(
        torq_compiler, tmp_path, grouped_k_module(kv=1), "torq-optimize-linalg-for-torq",
        "--torq-head-broadcast-mode=disable")
    assert "torq-broadcast-matmul" not in off, f"the flag-off control must not fold:\n{off}"
    assert "tensor<1x1x3x7x5xf32>" in off, f"the rank-5 chain must survive the control:\n{off}"
    assert "linalg.batch_matmul" in off, f"the matmul must survive the control:\n{off}"


def grouped_v_module(wrong_merge=False, kv=2):
    """One context matmul: probs @ repeat(v_cache). KV = 2, G = 3, H = 6,
    S = 7, D = 5. `wrong_merge` uses the [[0,1],[2,3],[4]] collapse over the
    rank-5 source: type-valid (the matmul's batch extent is KV, not H), but
    not the observed head merge."""
    g, s, d = 3, 7, 5
    h = kv * g
    if wrong_merge:
        reassoc = "[[0, 1], [2, 3], [4]]"
        b_shape = f"tensor<{kv}x{g * s}x{d}xf32>"
        a_k, batch = g * s, kv
    else:
        reassoc = "[[0, 1, 2], [3], [4]]"
        b_shape = f"tensor<{h}x{s}x{d}xf32>"
        a_k, batch = s, h
    return f"""
func.func @grouped_v(%p: tensor<{batch}x1x{a_k}xf32>, %v: tensor<1x{kv}x{s}x{d}xf32>) -> tensor<{batch}x1x{d}xf32> {{
  %zero = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x{kv}x{g}x{s}x{d}xf32>
  %1 = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}} ins(%v : tensor<1x{kv}x{s}x{d}xf32>) outs(%0 : tensor<1x{kv}x{g}x{s}x{d}xf32>) {{
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }} -> tensor<1x{kv}x{g}x{s}x{d}xf32>
  %c = tensor.collapse_shape %1 {reassoc} : tensor<1x{kv}x{g}x{s}x{d}xf32> into {b_shape}
  %f = tensor.empty() : tensor<{batch}x1x{d}xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%f : tensor<{batch}x1x{d}xf32>) -> tensor<{batch}x1x{d}xf32>
  %mm = linalg.batch_matmul ins(%p, %c : tensor<{batch}x1x{a_k}xf32>, {b_shape}) outs(%acc : tensor<{batch}x1x{d}xf32>) -> tensor<{batch}x1x{d}xf32>
  return %mm : tensor<{batch}x1x{d}xf32>
}}
"""


@pytest.mark.ci
@pytest.mark.parametrize(
    "variant",
    ["wrong_perm", "identity_perm", "m_gt_1", "g_eq_1", "forked_broadcast",
     "body_reads_init", "wrong_merge", "v_wrong_merge"],
)
def test_grouped_head_broadcast_near_misses(torq_compiler, tmp_path, variant):
    if variant == "v_wrong_merge":
        module = grouped_v_module(wrong_merge=True)
    else:
        args = {
            "wrong_perm": {"perm": (1, 0, 3, 2)},
            # The identity perm was value-safe under the pre-narrowing link
            # walk; the strict signature declines it (future relaxation).
            "identity_perm": {"perm": (0, 1, 2, 3)},
            "m_gt_1": {"m": 2},
            "g_eq_1": {"g": 1},
            "forked_broadcast": {"fork_broadcast": True},
            "body_reads_init": {"body_reads_init": True},
            "wrong_merge": {"wrong_merge": True},
        }[variant]
        module = grouped_k_module(**args)
    out = run_optimize_pass_standalone(
        torq_compiler, tmp_path, module, "torq-optimize-linalg-for-torq"
    )
    assert "tensor<2x3x5xf32>" not in out and "tensor<2x3x7xf32>" not in out, out
    assert "linalg.batch_matmul" in out, f"the matmul must survive {variant}:\n{out}"


@pytest.mark.ci
def test_grouped_head_broadcast_scale_from_argument(torq_compiler, tmp_path):
    module = grouped_k_module().replace(
        '%k: tensor<1x2x7x5xf32>)',
        '%k: tensor<1x2x7x5xf32>, %data: tensor<1x6x5x7xf32>)',
    ).replace('ins(%t, %scale', 'ins(%data, %scale')
    out = run_optimize_pass_standalone(
        torq_compiler, tmp_path, module, "torq-optimize-linalg-for-torq"
    )
    assert "tensor<2x3x5xf32>" not in out, out
    assert "linalg.batch_matmul" in out


@pytest.mark.ci
def test_grouped_head_broadcast_does_not_replay_rejected_operand(torq_compiler, tmp_path):
    # Both operands have head chains. Only the RHS cache may fold into M;
    # its replay links must not come from the LHS chain.
    key = grouped_k_module().replace('x5x', 'x1x')
    key_ops = key[key.index('  %scale'):key.index('  %f =')]
    key_ops = re.sub(r'%([a-zA-Z0-9_]+)', r'%key_\1', key_ops)
    module = grouped_v_module().replace(
        '%v: tensor<1x2x7x5xf32>)',
        '%v: tensor<1x2x7x5xf32>, %key_k: tensor<1x2x7x1xf32>)',
    ).replace('  %zero =', key_ops + '  %zero =', 1)
    module = module.replace('ins(%p, %c', 'ins(%key_c, %c')
    out = run_optimize_pass_standalone(
        torq_compiler, tmp_path, module, "torq-optimize-linalg-for-torq"
    )
    assert "tensor<2x3x7xf32>" in out
    assert "floordiv" not in out
    assert "tensor<1x2x5x7xf32>" not in out


@pytest.mark.ci
def test_grouped_head_broadcast_preserves_custom_matmul_maps(torq_compiler, tmp_path):
    # Equal extents keep a transposed RHS map type-valid; the operation is
    # different even though the operand shapes match the standard form.
    module = grouped_k_module().replace('x7x', 'x5x').replace(
        'linalg.batch_matmul ins',
        'linalg.batch_matmul indexing_maps = ['
        'affine_map<(b, m, n, k) -> (b, m, k)>, '
        'affine_map<(b, m, n, k) -> (b, n, k)>, '
        'affine_map<(b, m, n, k) -> (b, m, n)>] ins',
    )
    out = run_optimize_pass_standalone(
        torq_compiler, tmp_path, module, "torq-optimize-linalg-for-torq"
    )
    assert "tensor<2x3x5xf32>" not in out, out
    assert "linalg.batch_matmul" in out


@pytest.mark.ci
def test_grouped_head_broadcast_fires_on_onnx_fixture(torq_compiler, tmp_path):
    # Both score and context matmuls use the small cache with heads in M.
    model = onnx.load(ONNX_MODELS / "gqa_head_broadcast.onnx")
    on = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq")
    assert "floordiv" not in on, on
    assert "tensor<2x3x5xf32>" in on and "tensor<2x3x7xf32>" in on, on
    assert "tensor<1x2x3x7x5xf32>" not in on, f"the expanded cache is still materialized:\n{on}"
    off = dump_after(
        torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq",
        "--torq-head-broadcast-mode=disable")
    assert "torq-broadcast-matmul" not in off, \
        f"the disable mode must leave the chain in place:\n{off}"
    assert off.count("linalg.batch_matmul") == 2, f"both matmuls must survive:\n{off}"
    assert "tensor<1x6x5x7xf32>" in off, f"the full-size K^T must survive:\n{off}"


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


def pow_model(exponent, shape=(1, 196, 24)):
    """Pow(x, exponent) with a constant scalar exponent."""
    vi = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, list(shape))
    node = helper.make_node("Pow", ["x", "e"], ["y"])
    graph = helper.make_graph(
        [node], "pow", [vi("x")], [vi("y")],
        [numpy_helper.from_array(np.array(exponent, np.float32), "e")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


POW_OP = re.compile(r"math\.(fpowi|powf)\b")


@pytest.mark.ci
def test_pow3_to_mul_fires(torq_compiler, tmp_path):
    # bf16 x^3: PowToMulPattern expands it into mul(mul(x, x), x), so no pow op
    # is left for the host.
    pass_arg = "torq-optimize-linalg-for-torq"
    text = dump_after(torq_compiler, tmp_path, pow_model(3.0), pass_arg, *CONVERT_DTYPES,
                      f"--mlir-print-ir-before={pass_arg}")
    before, after = ir_dumps(text, "Before")[0], ir_dumps(text, "After")[0]
    assert POW_OP.search(before) and "bf16" in before, f"control: no bf16 pow to expand:\n{before}"
    assert not POW_OP.search(after), f"the pow op was not expanded:\n{after}"
    assert after.count("arith.mulf") == 2, f"x^3 is not two multiplications:\n{after}"


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


def run_optimize_pass_standalone(torq_compiler, tmp_path, module_text, pass_spec, *flags):
    """Run OptimizeLinalgForTorq via iree-opt on module text; return the IR.
    `flags` are extra global command-line options (e.g. a pattern mode flag)."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    mlir = tmp_path / "module.mlir"
    mlir.write_text(module_text)
    cmd = [str(iree_opt), str(mlir),
           "--pass-pipeline", f"builtin.module(func.func({pass_spec}))", *flags]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"iree-opt failed:\n{proc.stderr}"
    return proc.stdout


@pytest.mark.ci
@pytest.mark.parametrize("variant", ["sum", "extended", "commuted", "square", "duplicate", "permuted", "nonzero_init", "buffer"])
def test_generic_reduce_sum_host_routing(torq_compiler, tmp_path, variant):
    module = """
func.func @main(%a: tensor<2x3x8xf32>) -> tensor<2x3xf32> {
  %c = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x3xf32>
  %f = linalg.fill ins(%c : f32) outs(%e : tensor<2x3xf32>) -> tensor<2x3xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a : tensor<2x3x8xf32>) outs(%f : tensor<2x3xf32>) {
  ^bb0(%in: f32, %init: f32):
    %s = arith.addf %in, %init : f32
    linalg.yield %s : f32
  } -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}
"""
    if variant == "extended":
        module = module.replace("2x3x8xf32", "2x3x8xbf16").replace("%in: f32", "%in: bf16")
        module = module.replace("%s = arith.addf %in", "%x = arith.extf %in : bf16 to f32\n    %s = arith.addf %x")
    elif variant == "commuted":
        module = module.replace("arith.addf %in, %init", "arith.addf %init, %in")
    elif variant == "square":
        module = module.replace("%s = arith.addf %in", "%x = arith.mulf %in, %in : f32\n    %s = arith.addf %x")
    elif variant == "duplicate":
        module = module.replace("arith.addf %in, %init", "arith.addf %in, %in")
    elif variant == "permuted":
        module = module.replace("-> (d0, d1)>", "-> (d1, d0)>").replace("2x3xf32", "3x2xf32")
    elif variant == "nonzero_init":
        module = module.replace("constant 0.0", "constant 1.0")
    elif variant == "buffer":
        module = """
func.func @main(%a: memref<2x8xf32>, %out: memref<2xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%a : memref<2x8xf32>) outs(%out : memref<2xf32>) {
  ^bb0(%in: f32, %init: f32):
    %s = arith.addf %in, %init : f32
    linalg.yield %s : f32
  }
  return
}
"""
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, module, "torq-mark-host-executor")
    assert "linalg.generic" in out, out
    if variant == "buffer":
        return  # Host routing may see buffers; the lowering pass expects tensors.
    supported = variant in {"sum", "extended", "commuted"}
    assert ("#torq_hl.executor<host>" not in out) == supported, out
    # The lowering must use the same structural checks as host routing.
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, module, "torq-linalg-to-torqhl-conversion")
    assert ('"torq_hl.reduce"' in out) == supported, out


@pytest.mark.ci
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("body_op,identity", [("addf", "0.0"), ("maximumf", "0xFF800000")])
def test_generic_keepdims_reduction(torq_compiler, tmp_path, axis, body_op, identity):
    shape = [2, 3, 8]
    shape[axis] = 1
    output = "x".join(map(str, shape)) + "xf32"
    dims = ["d0", "d1", "d2"]
    dims[axis] = "0"
    iterators = ["parallel"] * 3
    iterators[axis] = "reduction"
    iterator_types = ", ".join(f'"{it}"' for it in iterators)
    module = f"""
func.func @main(%a: tensor<2x3x8xf32>) -> tensor<{output}> {{
  %c = arith.constant {identity} : f32
  %e = tensor.empty() : tensor<{output}>
  %f = linalg.fill ins(%c : f32) outs(%e : tensor<{output}>) -> tensor<{output}>
  %r = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ({', '.join(dims)})>], iterator_types = [{iterator_types}]}} ins(%a : tensor<2x3x8xf32>) outs(%f : tensor<{output}>) {{
  ^bb0(%in: f32, %init: f32):
    %s = arith.{body_op} %in, %init : f32
    linalg.yield %s : f32
  }} -> tensor<{output}>
  return %r : tensor<{output}>
}}
"""
    out = run_optimize_pass_standalone(
        torq_compiler, tmp_path, module,
        "torq-optimize-linalg-for-torq,torq-linalg-to-torqhl-conversion",
    )
    assert '"torq_hl.reduce"' in out, out
    assert "linalg.generic" not in out, out
    assert f"-> tensor<{output}>" in out, out


@pytest.mark.ci
def test_keepdims_reduction_keeps_other_unit_dims(torq_compiler, tmp_path):
    # Only the constant-zero output axis is dropped: the input and the batch of 1
    # keep their shape.
    module = """
func.func @main(%a: tensor<1x4x7x7xf32>) -> tensor<1x4x7x1xf32> {
  %c = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x4x7x1xf32>
  %f = linalg.fill ins(%c : f32) outs(%e : tensor<1x4x7x1xf32>) -> tensor<1x4x7x1xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a : tensor<1x4x7x7xf32>) outs(%f : tensor<1x4x7x1xf32>) {
  ^bb0(%in: f32, %init: f32):
    %s = arith.addf %in, %init : f32
    linalg.yield %s : f32
  } -> tensor<1x4x7x1xf32>
  return %r : tensor<1x4x7x1xf32>
}
"""
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, module, "torq-optimize-linalg-for-torq")
    assert "ins(%arg0 : tensor<1x4x7x7xf32>) outs(%collapsed : tensor<1x4x7xf32>)" in out, out
    assert "tensor.expand_shape" in out and "into tensor<1x4x7x1xf32>" in out, out


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
@pytest.mark.parametrize("custom_maps", [False, True])
def test_per_tensor_dequant_matmul_single_block_form(torq_compiler, tmp_path, custom_maps):
    module = DEQUANT_MODULE
    if custom_maps:
        module = module.replace("tensor<1x1x4xbf16>", "tensor<1x4x1xbf16>")
        module = module.replace("linalg.batch_matmul ins", "linalg.batch_matmul indexing_maps = ["
            "affine_map<(b,m,n,k)->(b,k,m)>, affine_map<(b,m,n,k)->(b,k,n)>, "
            "affine_map<(b,m,n,k)->(b,m,n)>] ins")
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, module, "torq-optimize-linalg-for-torq")
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


# --- Constant pad -> concat --------------------------------------------------
#
# PadToConcatPattern rewrites a pad that appends a scalar constant along a
# single axis into tensor.concat(source, constant-filled tensor), so the
# padded region is one constant insert instead of a fill-and-copy kernel.
# The ONNX fixture carries the three rewritable pads of a padded attention
# block (heads 5, S 11, D 36 -> 40; the K^T pad on its head-dim axis) plus two
# declined near-miss pads (multi-axis and leading); the near misses without an
# ONNX form are driven standalone below. An all-zero pad is a separate
# non-case: PadOp::fold canonicalizes it away before the pass runs, so the
# pattern (and its tests) only see pads with a non-zero amount.

PAD_PASS = "torq-optimize-linalg-for-torq"


@pytest.mark.ci
def test_pad_to_concat_fires_on_onnx_fixture(torq_compiler, tmp_path):
    model = onnx.load(ONNX_MODELS / "pad_to_concat.onnx")
    on = dump_after(torq_compiler, tmp_path, model, PAD_PASS)
    # The three single-axis constant pads become concats with zero-filled
    # tensors; only the multi-axis and the leading pad remain.
    assert on.count("tensor.pad") == 2, \
        f"only the two declined near-miss pads may remain:\n{on}"
    assert on.count("tensor.concat") == 3, f"each rewritable pad becomes a concat:\n{on}"
    for fill in ["tensor<1x5x1x4xf32>", "tensor<1x5x4x11xf32>", "tensor<1x5x11x4xf32>"]:
        assert fill in on, f"the constant fill {fill} is missing:\n{on}"
    # The disable control keeps all five pads and adds no concat.
    off = dump_after(torq_compiler, tmp_path, model, PAD_PASS, "--torq-pad-to-concat-mode=disable")
    assert off.count("tensor.pad") == 5, f"the disable control must keep every pad:\n{off}"
    assert off.count("tensor.concat") == 0, f"the disable control must add no concat:\n{off}"


@pytest.mark.ci
@pytest.mark.parametrize("op,kernel", [
    ("conv_2d_nchw_fchw", "tensor<8x8x3x3xf32>"),
    ("pooling_nchw_max", "tensor<3x3xf32>"),
])
def test_pad_to_concat_preserves_sliding_window_padding(torq_compiler, tmp_path, op, kernel):
    module = f"""
func.func @pad(%src: tensor<1x8x7x40xf32>, %kernel: {kernel},
               %init: tensor<1x8x3x20xf32>) -> tensor<1x8x3x20xf32> {{
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %src low[0, 0, 0, 0] high[0, 0, 0, 1] {{
  ^bb0(%n: index, %c: index, %h: index, %w: index):
    tensor.yield %zero : f32
  }} : tensor<1x8x7x40xf32> to tensor<1x8x7x41xf32>
  %result = linalg.{op} {{strides = dense<2> : tensor<2xi64>,
                          dilations = dense<1> : tensor<2xi64>}}
    ins(%padded, %kernel : tensor<1x8x7x41xf32>, {kernel})
    outs(%init : tensor<1x8x3x20xf32>) -> tensor<1x8x3x20xf32>
  return %result : tensor<1x8x3x20xf32>
}}
"""
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, module, PAD_PASS)
    assert "tensor.pad" in out, f"hardware input padding must remain visible:\n{out}"
    assert "tensor.concat" not in out, f"input padding must not become a concat:\n{out}"


def pad_to_concat_module(negative=False, position_dependent=False, dynamic=False):
    """One constant pad the concat rewrite targets (rank 2, 4x7 -> 4x9, the
    last axis extended by 2), with a single guard violated per flag: a
    negative low pad, a position-dependent (non-constant) value, or a dynamic
    shape. The result is returned, so each near miss must keep its pad."""
    if dynamic:
        src, res = "tensor<?x7xf32>", "tensor<?x9xf32>"
    elif negative:
        src, res = "tensor<4x7xf32>", "tensor<2x7xf32>"
    else:
        src, res = "tensor<4x7xf32>", "tensor<4x9xf32>"
    low = "[-2, 0]" if negative else "[0, 0]"
    high = "[0, 0]" if negative else "[0, 2]"
    extras = ("  %c1 = arith.constant 1.000000e+00 : f32\n  %c2 = arith.constant 2 : index\n"
              if position_dependent else "")
    body = (
        "%in_range = arith.cmpi \"slt\", %j, %c2 : index\n"
        "    %v = arith.select %in_range, %c1, %c0 : f32\n"
        "    tensor.yield %v : f32"
        if position_dependent
        else "tensor.yield %c0 : f32"
    )
    return f"""
func.func @pad(%src: {src}) -> {res} {{
  %c0 = arith.constant 0.000000e+00 : f32
{extras}  %padded = tensor.pad %src low{low} high{high} {{
  ^bb0(%i: index, %j: index):
    {body}
  }} : {src} to {res}
  return %padded : {res}
}}
"""


@pytest.mark.ci
@pytest.mark.parametrize("variant", ["negative_low", "position_dependent", "dynamic_shape"])
def test_pad_to_concat_near_misses(torq_compiler, tmp_path, variant):
    flags = {
        "negative_low": {"negative": True},
        "position_dependent": {"position_dependent": True},
        "dynamic_shape": {"dynamic": True},
    }[variant]
    out = run_optimize_pass_standalone(
        torq_compiler, tmp_path, pad_to_concat_module(**flags), PAD_PASS
    )
    assert "tensor.pad" in out, f"the near miss {variant} was rewritten:\n{out}"


@pytest.mark.ci
def test_pad_to_concat_disable_patterns_label(torq_compiler, tmp_path):
    # The default mode rewrites the single trailing pad; the pass-scoped label
    # disables it with the mode flag left at its default.
    module = pad_to_concat_module()
    on = run_optimize_pass_standalone(torq_compiler, tmp_path, module, PAD_PASS)
    assert "tensor.concat" in on and "tensor<4x2xf32>" in on, \
        f"the default mode must rewrite the pad:\n{on}"
    off = run_optimize_pass_standalone(
        torq_compiler, tmp_path, module,
        'torq-optimize-linalg-for-torq{disable-patterns="PadToConcatPattern"}')
    assert "tensor.pad" in off, f"the pass-scoped label must disable the rewrite:\n{off}"
    assert "tensor.concat" not in off, f"no concat may appear under the label:\n{off}"


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


# Since IREE 3.12, torch-mlir accumulates a bf16 matmul in f32 and truncates the result in a
# separate generic. FoldTruncfIntoFloatMatmulPattern folds the truncf into the matmul, so it
# writes bf16. The control is the IR before the pass, which still has the f32 matmul and truncf.

def bf16_const_weight_matmul(a_shape):
    """A @ W with W a constant [8,16] bf16 initializer. A [1,H,W,8] input is the form of #2675."""
    weight = np.random.default_rng(0).standard_normal((8, 16)) * 0.1
    node = helper.make_node("MatMul", ["a", "W"], ["o"], name="m")
    graph = helper.make_graph(
        [node], "bf16_const_weight_matmul",
        [helper.make_tensor_value_info("a", TensorProto.BFLOAT16, a_shape)],
        [helper.make_tensor_value_info("o", TensorProto.BFLOAT16, a_shape[:-1] + [16])],
        [float_initializer("W", weight, TensorProto.BFLOAT16)])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def dispatch_dump(text, when):
    """The dispatch dump printed `when` a pass, skipping the tile-fit probe's extracted copies."""
    dumps = [d for d in ir_dumps(text, when) if "_dispatch_" in d and "extracted_ops" not in d]
    assert dumps, f"no dispatch dump {when.lower()} the pass:\n{text}"
    return dumps[0]


@pytest.mark.ci
@pytest.mark.parametrize("make_model, matmul_op, absorbs_broadcast", [
    (lambda: bf16_const_weight_matmul([4, 8]), "linalg.matmul", False),
    (lambda: bf16_const_weight_matmul([1, 4, 4, 8]), "linalg.batch_matmul", False),
    # The batch matmul pattern also absorbs the key broadcast here, so the fold must not
    # depend on which of the two patterns the greedy driver applies first.
    (lambda: gqa_key_scores(dtype=TensorProto.BFLOAT16), "linalg.batch_matmul", True),
], ids=["const_weight_2d", "const_weight_4d", "gqa_key_broadcast"])
# runtime_hw_type is unused, but requesting it makes the plugin run the test for each chip.
def test_fold_truncf_into_float_matmul(torq_compiler, tmp_path, runtime_hw_type, chip_config,
                                       make_model, matmul_op, absorbs_broadcast):
    pass_arg = "torq-optimize-linalg-for-torq"
    # dump_after pins SL2610; the last --torq-hw wins.
    text = dump_after(torq_compiler, tmp_path, make_model(), pass_arg,
                      f"--mlir-print-ir-before={pass_arg}", f"--torq-hw={chip_config.data['target']}")
    before, after = dispatch_dump(text, "Before"), dispatch_dump(text, "After")
    f32_matmul = re.compile(rf"{re.escape(matmul_op)} .*-> tensor<[0-9x]+xf32>")
    assert f32_matmul.search(before) and "arith.truncf" in before, \
        f"control: no f32 {matmul_op} and truncf to fold:\n{before}"
    assert "xf32>" not in after and "arith.truncf" not in after, \
        f"the matmul still writes f32 or a separate truncf is left:\n{after}"
    if absorbs_broadcast:
        assert "torq-broadcast-matmul" in after, f"the key broadcast was not absorbed:\n{after}"


# --- Decomposed tanh GELU -----------------------------------------------------
#
# FuseDecomposedGeluPattern replaces the chain of one elementwise generic per
# ONNX node (Mul, Pow, Mul, Add, Mul, Tanh, Add, Mul) with the single GELU
# generic that GeluOpPattern lowers.


def decomposed_tanh_gelu(cube_as_pow, cube_coeff=0.044715, tanh_is_output=False):
    """0.5 * x * (1 + tanh(0.7978 * (x + 0.044715 * x^3))) in bf16, one node per
    step, as Redimnet2 exports it. x^3 is a Pow node, or Mul(Mul(x, x), x).
    With tanh_is_output, the tanh result is a second output of the model."""
    def scalar(name, value):
        # The importer reads bf16 initializers from raw data only
        raw = np.array(value, dtype=ml_dtypes.bfloat16).tobytes()
        return helper.make_tensor(name, TensorProto.BFLOAT16, [], raw, raw=True)

    if cube_as_pow:
        cube_nodes = [helper.make_node("Pow", ["x", "three"], ["cube"])]
    else:
        cube_nodes = [helper.make_node("Mul", ["x", "x"], ["square"]),
                      helper.make_node("Mul", ["square", "x"], ["cube"])]
    nodes = [
        helper.make_node("Mul", ["x", "half"], ["half_x"]),
        *cube_nodes,
        helper.make_node("Mul", ["cube", "coeff"], ["cube_term"]),
        helper.make_node("Add", ["x", "cube_term"], ["sum"]),
        helper.make_node("Mul", ["sum", "sqrt_2_over_pi"], ["tanh_in"]),
        helper.make_node("Tanh", ["tanh_in"], ["tanh_out"]),
        helper.make_node("Add", ["tanh_out", "one"], ["one_plus_tanh"]),
        helper.make_node("Mul", ["half_x", "one_plus_tanh"], ["y"]),
    ]
    shape = [1, 196, 24]
    graph = helper.make_graph(
        nodes, "decomposed_tanh_gelu",
        [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, shape)],
        [helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape)
         for name in ["y"] + (["tanh_out"] if tanh_is_output else [])],
        [scalar("half", 0.5), scalar("three", 3.0), scalar("coeff", cube_coeff),
         scalar("sqrt_2_over_pi", 0.7978845608), scalar("one", 1.0)])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.ci
@pytest.mark.parametrize("cube_as_pow", [True, False])
def test_decomposed_tanh_gelu_fuses_to_one_generic(torq_compiler, tmp_path, cube_as_pow):
    pass_arg = "torq-optimize-linalg-for-torq"
    text = dump_after(torq_compiler, tmp_path, decomposed_tanh_gelu(cube_as_pow), pass_arg,
                      f"--mlir-print-ir-before={pass_arg}")
    [before] = ir_dumps(text, "Before")
    [after] = ir_dumps(text, "After")
    # Control: the importer gives one generic per step
    assert count_ops(before, "linalg.generic") >= 8, f"control: no decomposed GELU:\n{before}"
    assert count_ops(after, "linalg.generic") == 1, f"the GELU chain was not fused:\n{after}"
    assert count_ops(after, "math.tanh") == 1, after
    assert count_ops(after, "math.fpowi") == 1, after


@pytest.mark.ci
@pytest.mark.parametrize("model_args", [
    # 0.05 instead of 0.044715 is outside the pattern's 1% tolerance: not a GELU
    dict(cube_coeff=0.05),
    # The tanh step would stay alive for its other user and be computed twice
    dict(tanh_is_output=True),
], ids=["other_constant", "step_with_other_user"])
def test_decomposed_tanh_gelu_is_not_fused(torq_compiler, tmp_path, model_args):
    pass_arg = "torq-optimize-linalg-for-torq"
    text = dump_after(torq_compiler, tmp_path, decomposed_tanh_gelu(True, **model_args), pass_arg,
                      f"--mlir-print-ir-before={pass_arg}")
    [before] = ir_dumps(text, "Before")
    [after] = ir_dumps(text, "After")
    assert count_ops(before, "linalg.generic") >= 8, f"control: no decomposed chain:\n{before}"
    assert count_ops(after, "linalg.generic") > 1 and count_ops(after, "math.tanh") == 1, \
        f"the chain was fused:\n{after}"

@pytest.mark.ci
@pytest.mark.parametrize("shape", [[1, 2], [1, 8, 1, 207], [1, 8, 207, 207], [8, 300, 300], [8, 400, 400]])
@pytest.mark.parametrize("convert_dtypes", [False, True])
def test_softmax_raised_before_keepdims_folding(torq_compiler, tmp_path, shape, convert_dtypes):
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Softmax", ["x"], ["y"], axis=-1)],
            "softmax",
            [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, shape)],
            [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, shape)],
        ),
        opset_imports=[helper.make_opsetid("", 22)],
    )
    flags = ["--torq-disable-host", "--torq-disable-css", "--torq-enable-profiling",
             "--mlir-print-ir-before=torq-optimize-linalg-for-torq"]
    if convert_dtypes:
        flags += ["--torq-convert-dtypes", "--torq-convert-io-dtype"]
    dump = dump_after(torq_compiler, tmp_path, model, "torq-optimize-linalg-for-torq", *flags)
    before = "\n".join(ir_dumps(dump, "Before"))
    after = "\n".join(ir_dumps(dump, "After"))
    assert "linalg.generic" in before, before
    assert "linalg.softmax" in after, after


@pytest.mark.ci
@pytest.mark.parametrize("lifted", [False, True])
def test_pad_to_concat_preserves_lifted_conv1d_padding(torq_compiler, tmp_path, lifted):
    module = """
func.func @pad(%src: tensor<1x8x40xf32>, %kernel: tensor<8x8x2xf32>,
               %init: tensor<1x8x40xf32>) -> tensor<1x8x40xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %src low[0, 0, 0] high[0, 0, 1] {
  ^bb0(%n: index, %c: index, %w: index):
    tensor.yield %zero : f32
  } : tensor<1x8x40xf32> to tensor<1x8x41xf32>
  %result = linalg.conv_1d_ncw_fcw {strides = dense<1> : tensor<1xi64>,
                                  dilations = dense<1> : tensor<1xi64>}
    ins(%padded, %kernel : tensor<1x8x41xf32>, tensor<8x8x2xf32>)
    outs(%init : tensor<1x8x40xf32>) -> tensor<1x8x40xf32>
  return %result : tensor<1x8x40xf32>
}
"""
    if lifted:
        # Exercise the same pad after Conv1D lowering has already inserted a unit axis.
        module = module.replace("  %result =", """
  %expanded = tensor.expand_shape %padded [[0], [1], [2, 3]] output_shape [1, 8, 1, 41] : tensor<1x8x41xf32> into tensor<1x8x1x41xf32>
  %kernel2 = tensor.expand_shape %kernel [[0], [1], [2, 3]] output_shape [8, 8, 1, 2] : tensor<8x8x2xf32> into tensor<8x8x1x2xf32>
  %init2 = tensor.expand_shape %init [[0], [1], [2, 3]] output_shape [1, 8, 1, 40] : tensor<1x8x40xf32> into tensor<1x8x1x40xf32>
  %result2 =""").replace("linalg.conv_1d_ncw_fcw", "linalg.conv_2d_nchw_fchw").replace("tensor<1xi64>", "tensor<2xi64>")
        module = module.replace("ins(%padded, %kernel : tensor<1x8x41xf32>, tensor<8x8x2xf32>)", "ins(%expanded, %kernel2 : tensor<1x8x1x41xf32>, tensor<8x8x1x2xf32>)")
        module = module.replace("outs(%init : tensor<1x8x40xf32>) -> tensor<1x8x40xf32>", "outs(%init2 : tensor<1x8x1x40xf32>) -> tensor<1x8x1x40xf32>\n  %result = tensor.collapse_shape %result2 [[0], [1], [2, 3]] : tensor<1x8x1x40xf32> into tensor<1x8x40xf32>")
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, module, PAD_PASS)
    assert "tensor.pad" in out, out
    assert "tensor.concat" not in out, out


@pytest.mark.ci
@pytest.mark.parametrize("kv", [1, 2])
@pytest.mark.parametrize("form", ["k", "v"])
@pytest.mark.parametrize("external_init", [False, True])
def test_grouped_heads_lower_to_torqhl_matmul(torq_compiler, tmp_path, kv, form, external_init):
    module = grouped_k_module(kv=kv) if form == "k" else grouped_v_module(kv=kv)
    module = module.replace("f32", "bf16")
    if external_init:
        width = 7 if form == "k" else 5
        module = module.replace(") ->", f", %init: tensor<{kv * 3}x1x{width}xbf16>) ->", 1)
        module = module.replace("outs(%acc :", "outs(%init :")
    out = run_optimize_pass_standalone(torq_compiler, tmp_path, module,
        "torq-optimize-linalg-for-torq,torq-linalg-to-torqhl-conversion")
    assert "torq_hl.matmul" in out, out
    if external_init:
        assert "linalg.fill" not in out, out
    assert "floordiv" not in out and "torq-broadcast-matmul" not in out, out


@pytest.mark.parametrize("kv,g", [(1, 3), (2, 3), (4, 8)])
@pytest.mark.parametrize("k,n", [(5, 7), (7, 5), (32, 64)])
def test_grouped_heads_preserve_matmul_and_accumulator(kv, g, k, n):
    rng = np.random.default_rng(0)
    lhs = rng.normal(size=(kv * g, 1, k))
    rhs = rng.normal(size=(kv, k, n))
    init = rng.normal(size=(kv * g, 1, n))
    expected = init + lhs @ np.repeat(rhs, g, axis=0)
    actual = (init.reshape(kv, g, n) + lhs.reshape(kv, g, k) @ rhs).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
