// Native multi-channel Conv1D (C=48, F=48, Kw=3), NSS-only. Covers the f32-output path
// (no truncf epilogue); filter is a runtime argument.
module {
  func.func @main(
      %arg0: tensor<1x48x35xbf16>,
      %arg1: tensor<48x48x3xbf16>
  ) -> tensor<1x48x33xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x48x33xf32>
    %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
        ins(%arg0, %arg1 : tensor<1x48x35xbf16>, tensor<48x48x3xbf16>)
        outs(%cst : tensor<1x48x33xf32>) -> tensor<1x48x33xf32>
    return %0 : tensor<1x48x33xf32>
  }
}
