// Constant-weight dilated conv2d (dilation [2, 2], k 3x3, expanded 5x5 in
// range) with a bf16 output.  Its output bytes x input channels (4.5 KB) is
// below the 1 MB crossover of the bf16/f16 dilation gate, so it keeps the
// native zero-interleaved form (the residue split's fixed costs dominate at
// this size); the llvm-cpu reference accumulates bf16 in bf16, so compare against the
// fp32-accumulating numpy reference (sweepconv-bf16 routing).
// TORQ_FP_MAX_TOL: 0.05
module {
  func.func @main(%0 : tensor<1x4x16x16xbf16>) -> tensor<1x4x12x12xbf16> {
    %1 = arith.constant dense<0.01> : tensor<4x4x3x3xbf16>
    %2 = arith.constant dense<0.000000e+00> : tensor<1x4x12x12xbf16>
    %3 = linalg.conv_2d_nchw_fchw {dilations = dense<[2, 2]> : vector<2xi64>, strides = dense<[1, 1]> : vector<2xi64>} ins(%0, %1 : tensor<1x4x16x16xbf16>, tensor<4x4x3x3xbf16>) outs(%2 : tensor<1x4x12x12xbf16>) -> tensor<1x4x12x12xbf16>
    return %3 : tensor<1x4x12x12xbf16>
  }
}
