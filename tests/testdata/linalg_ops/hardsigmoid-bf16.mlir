// ONNX HardSigmoid (alpha=1/6, beta=0.5) decomposed exactly the way the
// torch importer emits it: a bf16 multiply by a truncf'd captured f32
// scalar, an add by a captured bf16 scalar, and min/max clamps whose bounds
// arrive as 0-d constant tensor inputs. MulOpPattern's forward float tail
// fold must absorb the whole four-generic chain (mul -> add -> min -> max)
// into a single torq_hl.mul (ALU scalar-mul + ACT float bias + ACT float
// clip), so the dispatch stays one NSS kernel. Without the fold the
// mul/min/max generics have no lowering and test_linalg_ops (NSS-only)
// fails loudly.
//
// The fused op rounds once after the bias add, while the reference chain
// rounds per op: the difference is at most 1 bf16 ULP, which only shows as a
// large *relative* diff for outputs near the clamp boundary. Gate with an
// absolute tolerance (bf16 ULP at 1.0 is ~0.0039).
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.005
module {
  func.func public @main(%arg0: tensor<1x48x1x1xbf16>) -> tensor<1x48x1x1xbf16> {
    %cst_alpha = arith.constant 0.166666701 : f32
    %cst_beta = arith.constant 5.000000e-01 : bf16
    %cst_lo = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %cst_hi = arith.constant dense<1.000000e+00> : tensor<1xbf16>
    %e = tensor.empty() : tensor<1x48x1x1xbf16>
    %m = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x48x1x1xbf16>) outs(%e : tensor<1x48x1x1xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      %a = arith.truncf %cst_alpha : f32 to bf16
      %r = arith.mulf %in, %a : bf16
      linalg.yield %r : bf16
    } -> tensor<1x48x1x1xbf16>
    %a = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%m : tensor<1x48x1x1xbf16>) outs(%e : tensor<1x48x1x1xbf16>) {
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
