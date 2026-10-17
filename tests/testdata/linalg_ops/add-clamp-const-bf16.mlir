// A scalar-const add followed by two one-sided clamps (the torch bound-first
// ins(%bound, %data) form) is the hard-sigmoid chain without the multiply.
// AddOpPattern's forward float tail fold must absorb the add + min + max into
// a single torq_hl.add (ACT float bias + ACT float clip), so the dispatch
// stays one NSS kernel. The HW add lowering already applies
// act.rescaleClamp with these bounds; without the fold the add/min/max
// generics have no single-kernel lowering and this case would need three.
//
// The ACT rounds x + beta once (f32 bias, bf16 result), matching the per-op
// reference to within 1 bf16 ULP, so gate with an absolute tolerance.
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.005
module {
  func.func public @main(%arg0: tensor<1x48x1x1xbf16>) -> tensor<1x48x1x1xbf16> {
    %cst_beta = arith.constant 5.000000e-01 : bf16
    %cst_lo = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %cst_hi = arith.constant dense<1.000000e+00> : tensor<1xbf16>
    %e = tensor.empty() : tensor<1x48x1x1xbf16>
    %a = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x48x1x1xbf16>) outs(%e : tensor<1x48x1x1xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      %r = arith.addf %in, %cst_beta : bf16
      linalg.yield %r : bf16
    } -> tensor<1x48x1x1xbf16>
    %mi = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_hi, %a : tensor<1xbf16>, tensor<1x48x1x1xbf16>) outs(%e : tensor<1x48x1x1xbf16>) {
    ^bb0(%b: bf16, %in: bf16, %init: bf16):
      %c = arith.cmpf olt, %in, %b : bf16
      %r = arith.select %c, %in, %b : bf16
      linalg.yield %r : bf16
    } -> tensor<1x48x1x1xbf16>
    %ma = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_lo, %mi : tensor<1xbf16>, tensor<1x48x1x1xbf16>) outs(%e : tensor<1x48x1x1xbf16>) {
    ^bb0(%b: bf16, %in: bf16, %init: bf16):
      %c = arith.cmpf ogt, %in, %b : bf16
      %r = arith.select %c, %in, %b : bf16
      linalg.yield %r : bf16
    } -> tensor<1x48x1x1xbf16>
    return %ma : tensor<1x48x1x1xbf16>
  }
}
