// Compile regression: a scalar add inside a dual-sliced MatMul+GELU+LayerNorm
// chain on a 288x207 activation whose row width (207) is not a multiple of the
// bf16 vector width (8).
//
// The chain is large enough for the buffer planner to keep the intermediates in
// LRAM, and LinalgSlicing dual-slices it across the two NPU slices (288 rows ->
// two 144-row chunks), so the scalar adds run in place on packed strided
// 144-row subviews (offset 0 and 29808) of the shared 288-row LRAM buffers.
// The shape mirrors the moonshine encoder's first GEMM+GELU+LayerNorm
// (288x207 GEMM output, K=1728).
//
// History: a tile-and-fuse rework routed these packed strided subviews to the
// non-dense scalar builder (isDenseInMemory only recognizes identity layouts),
// which at the time tiled each row in div_ceil(207, 8) = 26 blocks of 8 items,
// modeling a 208-item row. The hardware writes whole 16-byte blocks, so the
// phantom tail items of each row's last block were written two bytes past the
// row and, for the last row, past the end of the 59616-byte view; the DEQW
// bounds check (SliceTaskOp::verify) then rejected the dispatch (maximum
// address 59617 >= 59616) and compilation of the moonshine encoder failed.
// The tiling hazard is gone from the non-dense builder (it now splits rows
// into equal exact blocks, #2702), but the misrouting remains: without this
// fix the adds below lower as 69 blocks of 3 items per row, instead of the
// efficient single sweep. The fix routes physically contiguous inputs
// (identity or packed strided subviews) with a contiguous output to the
// dense/linear builder, whose DEQW uses the XY view (A(H) [3728] + X(S)
// [29808] here) that discards the unaligned tail — the NDL form the
// pre-regression build emitted.
//
// This case is compile-only (see tests/test_linalg_compilation.py): its
// numerics are not validatable within the suite's tolerances on the available
// reference paths. Running it on the NPU sim yields valid results (no
// inf/NaN) but carries a bf16 drift (a few % of elements past relative 0.2,
// dominated by long-K matmul accumulation), and the llvm-cpu backend's bf16
// matmul accumulates differently from the NPU's fp32 accumulation for long K
// (visible with the matmul alone), so it is not a bit-reliable reference for
// this pattern either. The moonshine encoder full model covers the real model,
// but its sim run is xfailed because the CModel times out
// (synaptics-torq/torq-compiler-dev#1994).

module {
  func.func @main(%act: tensor<1728x207xbf16>, %w: tensor<288x1728xbf16>) -> tensor<288x207xbf16> {
    %scalar1 = arith.constant dense<0.5> : tensor<bf16>
    %scalar2 = arith.constant dense<0.25> : tensor<bf16>
    %c0 = arith.constant 0.0 : bf16
    %mm_e = tensor.empty() : tensor<288x207xbf16>
    %mm_init = linalg.fill ins(%c0 : bf16) outs(%mm_e : tensor<288x207xbf16>) -> tensor<288x207xbf16>
    %mm = linalg.matmul ins(%w, %act : tensor<288x1728xbf16>, tensor<1728x207xbf16>) outs(%mm_init : tensor<288x207xbf16>) -> tensor<288x207xbf16>
    %mul1_e = tensor.empty() : tensor<288x207xbf16>
    %mul1 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%mm, %mm : tensor<288x207xbf16>, tensor<288x207xbf16>) outs(%mul1_e : tensor<288x207xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %m = arith.mulf %a, %b : bf16
      linalg.yield %m : bf16
    } -> tensor<288x207xbf16>
    %erf_e = tensor.empty() : tensor<288x207xbf16>
    %erf = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%mul1 : tensor<288x207xbf16>) outs(%erf_e : tensor<288x207xbf16>) {
    ^bb0(%a: bf16, %out: bf16):
      %r = math.erf %a : bf16
      linalg.yield %r : bf16
    } -> tensor<288x207xbf16>
    %add_e = tensor.empty() : tensor<288x207xbf16>
    %add = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> ()>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%erf, %scalar1 : tensor<288x207xbf16>, tensor<bf16>) outs(%add_e : tensor<288x207xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %s2 = arith.addf %a, %b : bf16
      linalg.yield %s2 : bf16
    } -> tensor<288x207xbf16>
    %gelu_e = tensor.empty() : tensor<288x207xbf16>
    %gelu = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%mm, %add : tensor<288x207xbf16>, tensor<288x207xbf16>) outs(%gelu_e : tensor<288x207xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %m = arith.mulf %a, %b : bf16
      linalg.yield %m : bf16
    } -> tensor<288x207xbf16>
    %rm_e = tensor.empty() : tensor<207xbf16>
    %rm_init = linalg.fill ins(%c0 : bf16) outs(%rm_e : tensor<207xbf16>) -> tensor<207xbf16>
    %rm = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>
        ],
        iterator_types = ["reduction", "parallel"]
    } ins(%gelu : tensor<288x207xbf16>) outs(%rm_init : tensor<207xbf16>) {
    ^bb0(%a: bf16, %out: bf16):
      %s = arith.addf %a, %out : bf16
      linalg.yield %s : bf16
    } -> tensor<207xbf16>
    %sub_e = tensor.empty() : tensor<288x207xbf16>
    %sub = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%gelu, %rm : tensor<288x207xbf16>, tensor<207xbf16>) outs(%sub_e : tensor<288x207xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %s = arith.subf %a, %b : bf16
      linalg.yield %s : bf16
    } -> tensor<288x207xbf16>
    %sca_e = tensor.empty() : tensor<288x207xbf16>
    %sca = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> ()>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%sub, %scalar2 : tensor<288x207xbf16>, tensor<bf16>) outs(%sca_e : tensor<288x207xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %s2 = arith.addf %a, %b : bf16
      linalg.yield %s2 : bf16
    } -> tensor<288x207xbf16>
    return %sca : tensor<288x207xbf16>
  }
}
