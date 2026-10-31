module {
  func.func @main(%62 : tensor<1x2x5x13xbf16>, %61 : tensor<3x2x3x3xbf16>) -> tensor<1x3x3x3xbf16> {
    %63 = arith.constant dense<0.000000e+00> : tensor<1x3x3x3xbf16>
    %64 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 3]> : vector<2xi64>,   strides = dense<[1, 3]> : vector<2xi64>} ins(%62, %61 : tensor<1x2x5x13xbf16>, tensor<3x2x3x3xbf16>) outs(%63 : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
    return %64 : tensor<1x3x3x3xbf16>
  }
}
