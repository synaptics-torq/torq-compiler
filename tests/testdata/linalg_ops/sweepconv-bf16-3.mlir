module {
  func.func @main(%28 : tensor<1x2x9x9xbf16>, %27 : tensor<3x2x3x3xbf16>) -> tensor<1x3x3x3xbf16> {
    %29 = arith.constant dense<0.000000e+00> : tensor<1x3x3x3xbf16>
    %30 = linalg.conv_2d_nchw_fchw {dilations = dense<[3, 1]> : vector<2xi64>,   strides = dense<[1, 3]> : vector<2xi64>} ins(%28, %27 : tensor<1x2x9x9xbf16>, tensor<3x2x3x3xbf16>) outs(%29 : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
    return %30 : tensor<1x3x3x3xbf16>
  }
}
