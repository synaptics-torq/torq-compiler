// TORQ_FP_MAX_TOL: 0.05
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.01
// The llvmcpu reference is less accurate than the NPU here (up to ~200 vs ~60 absolute error
// against a float64 matmul): the gate ignores near-zero outputs where this is large relatively.
module {
  func.func @main(%arg0: tensor<2x113x80xbf16>, %arg1: tensor<2x7x113xbf16>) -> tensor<2x7x80xbf16> {
    %cst = arith.constant dense<0.0> : tensor<1xbf16>
    %cst_0 = arith.constant dense<0.0> : tensor<1xbf16>
    %0 = tosa.matmul %arg1, %arg0, %cst, %cst_0 : (tensor<2x7x113xbf16>, tensor<2x113x80xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<2x7x80xbf16>
    return %0 : tensor<2x7x80xbf16>
  }
}
