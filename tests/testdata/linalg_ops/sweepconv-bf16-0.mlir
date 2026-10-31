module {
  func.func @main(%1 : tensor<1x2x5x5xbf16>, %0 : tensor<3x2x3x3xbf16>) -> tensor<1x3x3x3xbf16> {
    %2 = arith.constant dense<0.000000e+00> : tensor<1x3x3x3xbf16>
    %3 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : vector<2xi64>,   strides = dense<[1, 1]> : vector<2xi64>} ins(%1, %0 : tensor<1x2x5x5xbf16>, tensor<3x2x3x3xbf16>) outs(%2 : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
    return %3 : tensor<1x3x3x3xbf16>
  }
}
