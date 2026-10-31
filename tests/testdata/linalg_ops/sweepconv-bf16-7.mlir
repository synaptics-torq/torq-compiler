module {
  func.func @main(%112 : tensor<1x2x9x13xbf16>, %111 : tensor<3x2x3x3xbf16>) -> tensor<1x3x3x3xbf16> {
    %113 = arith.constant dense<0.000000e+00> : tensor<1x3x3x3xbf16>
    %114 = linalg.conv_2d_nchw_fchw {dilations = dense<[3, 3]> : vector<2xi64>,   strides = dense<[1, 3]> : vector<2xi64>} ins(%112, %111 : tensor<1x2x9x13xbf16>, tensor<3x2x3x3xbf16>) outs(%113 : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
    return %114 : tensor<1x3x3x3xbf16>
  }
}
