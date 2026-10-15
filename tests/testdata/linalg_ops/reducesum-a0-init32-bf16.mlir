// TORQ_FP_MAX_TOL: 0.05
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.05
// Reduces axis 0, so the kernel has no batch dim to index the init bias by.
// The llvmcpu reference sums in bf16 from the init, so an output near zero can
// be off by a large relative amount. An output counts as wrong only if it is
// also off by more than 5% of the largest output. A dropped init (32) is.
module {
  func.func @main(%a: tensor<8x16xbf16>) -> tensor<16xbf16> {
    %c = arith.constant 32.0 : bf16
    %e = tensor.empty() : tensor<16xbf16>
    %f = linalg.fill ins(%c : bf16) outs(%e : tensor<16xbf16>) -> tensor<16xbf16>
    %r = linalg.reduce ins(%a : tensor<8x16xbf16>) outs(%f : tensor<16xbf16>) dimensions = [0]
      (%in: bf16, %init: bf16) {
        %s = arith.addf %in, %init : bf16
        linalg.yield %s : bf16
      }
    return %r : tensor<16xbf16>
  }
}
