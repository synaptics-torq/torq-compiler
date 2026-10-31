// TORQ_FP_MAX_TOL: 0.05
module {
  func.func @main(%0 : tensor<1x2x9x15xbf16>, %1 : tensor<3x2x3x3xbf16>) -> tensor<1x3x4x5xbf16> {
    %2 = arith.constant dense<0.000000e+00> : tensor<1x3x4x5xbf16>
    %3 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : vector<2xi64>, strides = dense<[2, 3]> : vector<2xi64>} ins(%0, %1 : tensor<1x2x9x15xbf16>, tensor<3x2x3x3xbf16>) outs(%2 : tensor<1x3x4x5xbf16>) -> tensor<1x3x4x5xbf16>
    return %3 : tensor<1x3x4x5xbf16>
  }
}
