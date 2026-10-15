// TORQ_FP_MAX_TOL: 0.05
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.05
// The dims are not contiguous, so the reduce is a chain and the init is merged
// in its last step. The llvmcpu reference sums in bf16 from the init, so an
// output near zero can be off by a large relative amount. An output counts as
// wrong only if it is also off by more than 5% of the largest output. A dropped
// init (32) is.
module {
  func.func @main(%a: tensor<4x8x4xbf16>) -> tensor<8xbf16> {
    %c = arith.constant 32.0 : bf16
    %e = tensor.empty() : tensor<8xbf16>
    %f = linalg.fill ins(%c : bf16) outs(%e : tensor<8xbf16>) -> tensor<8xbf16>
    %r = linalg.reduce ins(%a : tensor<4x8x4xbf16>) outs(%f : tensor<8xbf16>) dimensions = [0, 2]
      (%in: bf16, %init: bf16) {
        %s = arith.addf %in, %init : bf16
        linalg.yield %s : bf16
      }
    return %r : tensor<8xbf16>
  }
}
