// TORQ_FP_MAX_TOL: 0.7
// TORQ_EPSILON: 1e-3
module {
  func.func @main(%arg0: tensor<2x127x80xbf16>, %arg1: tensor<2x3x127xbf16>) -> tensor<2x3x80xbf16> {
    %cst = arith.constant dense<0.0> : tensor<1xbf16>
    %cst_0 = arith.constant dense<0.0> : tensor<1xbf16>
    %0 = tosa.matmul %arg1, %arg0, %cst, %cst_0 : (tensor<2x3x127xbf16>, tensor<2x127x80xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<2x3x80xbf16>
    return %0 : tensor<2x3x80xbf16>
  }
}
