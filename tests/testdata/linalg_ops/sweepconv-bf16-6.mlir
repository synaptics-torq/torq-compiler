module {
  func.func @main(%71 : tensor<1x2x9x9xbf16>, %70 : tensor<3x2x3x3xbf16>) -> tensor<1x3x3x3xbf16> {
    %72 = arith.constant dense<0.000000e+00> : tensor<1x3x3x3xbf16>
    %73 = linalg.conv_2d_nchw_fchw {dilations = dense<[3, 3]> : vector<2xi64>,   strides = dense<[1, 1]> : vector<2xi64>} ins(%71, %70 : tensor<1x2x9x9xbf16>, tensor<3x2x3x3xbf16>) outs(%72 : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
    return %73 : tensor<1x3x3x3xbf16>
  }
}
