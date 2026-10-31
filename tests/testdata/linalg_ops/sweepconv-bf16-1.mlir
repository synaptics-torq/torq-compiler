// TORQ_FP_MAX_TOL: 0.021
module {
  func.func @main(%6 : tensor<1x2x5x9xbf16>, %5 : tensor<3x2x3x3xbf16>) -> tensor<1x3x3x3xbf16> {
    %7 = arith.constant dense<0.000000e+00> : tensor<1x3x3x3xbf16>
    %8 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : vector<2xi64>,   strides = dense<[1, 3]> : vector<2xi64>} ins(%6, %5 : tensor<1x2x5x9xbf16>, tensor<3x2x3x3xbf16>) outs(%7 : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
    return %8 : tensor<1x3x3x3xbf16>
  }
}
