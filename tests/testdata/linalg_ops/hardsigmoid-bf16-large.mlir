// Hard sigmoid chain at a size that TileAndFuse tiles and LinalgSlicing
// slices (scf.forall over the coalesced chain); MulOpPattern's forward
// float tail fold must fuse the four per-tile generics into one
// torq_hl.mul inside the loop.
// Same single-rounding vs per-op-rounding tolerance gate as
// hardsigmoid-bf16.mlir.
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.005
module {
  func.func public @main(%arg0: tensor<1x1024x64xbf16>) -> tensor<1x1024x64xbf16> {
    %cst_alpha = arith.constant 0.166666701 : f32
    %cst_beta = arith.constant 5.000000e-01 : bf16
    %cst_lo = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %cst_hi = arith.constant dense<1.000000e+00> : tensor<1xbf16>
    %e = tensor.empty() : tensor<1x1024x64xbf16>
    %m = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1024x64xbf16>) outs(%e : tensor<1x1024x64xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      %a = arith.truncf %cst_alpha : f32 to bf16
      %r = arith.mulf %in, %a : bf16
      linalg.yield %r : bf16
    } -> tensor<1x1024x64xbf16>
    %a = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%m : tensor<1x1024x64xbf16>) outs(%e : tensor<1x1024x64xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      %r = arith.addf %in, %cst_beta : bf16
      linalg.yield %r : bf16
    } -> tensor<1x1024x64xbf16>
    %mi = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_hi, %a : tensor<1xbf16>, tensor<1x1024x64xbf16>) outs(%e : tensor<1x1024x64xbf16>) {
    ^bb0(%b: bf16, %in: bf16, %init: bf16):
      %c = arith.cmpf olt, %in, %b : bf16
      %r = arith.select %c, %in, %b : bf16
      linalg.yield %r : bf16
    } -> tensor<1x1024x64xbf16>
    %ma = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_lo, %mi : tensor<1xbf16>, tensor<1x1024x64xbf16>) outs(%e : tensor<1x1024x64xbf16>) {
    ^bb0(%b: bf16, %in: bf16, %init: bf16):
      %c = arith.cmpf ogt, %in, %b : bf16
      %r = arith.select %c, %in, %b : bf16
      linalg.yield %r : bf16
    } -> tensor<1x1024x64xbf16>
    return %ma : tensor<1x1024x64xbf16>
  }
}
