module {
  func.func @main(%11 : tensor<1x2x9x5xbf16>, %10 : tensor<3x2x3x3xbf16>) -> tensor<1x3x3x3xbf16> {
    %12 = arith.constant dense<0.000000e+00> : tensor<1x3x3x3xbf16>
    %13 = linalg.conv_2d_nchw_fchw {dilations = dense<[3, 1]> : vector<2xi64>,   strides = dense<[1, 1]> : vector<2xi64>} ins(%11, %10 : tensor<1x2x9x5xbf16>, tensor<3x2x3x3xbf16>) outs(%12 : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
    return %13 : tensor<1x3x3x3xbf16>
  }
}
