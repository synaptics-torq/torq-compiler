// TORQ_FP_MAX_TOL: 0.05
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.05
// The llvmcpu reference sums in bf16 from the init, so an output near zero can
// be off by a large relative amount. An output counts as wrong only if it is
// also off by more than 5% of the largest output. A dropped init (32 / 8) is.
module {
  func.func @main(%a: tensor<8x8xbf16>) -> tensor<8xbf16> {
    %c = arith.constant 32.0 : bf16
    %n = arith.constant 8.0 : bf16
    %e = tensor.empty() : tensor<8xbf16>
    %f = linalg.fill ins(%c : bf16) outs(%e : tensor<8xbf16>) -> tensor<8xbf16>
    %s = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%a : tensor<8x8xbf16>) outs(%f : tensor<8xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %x = arith.addf %in, %out : bf16
      linalg.yield %x : bf16
    } -> tensor<8xbf16>
    %m = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%s : tensor<8xbf16>) outs(%e : tensor<8xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %y = arith.divf %in, %n : bf16
      linalg.yield %y : bf16
    } -> tensor<8xbf16>
    return %m : tensor<8xbf16>
  }
}
