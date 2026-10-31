module {
  func.func @main(%45 : tensor<1x2x5x9xbf16>, %44 : tensor<3x2x3x3xbf16>) -> tensor<1x3x3x3xbf16> {
    %46 = arith.constant dense<0.000000e+00> : tensor<1x3x3x3xbf16>
    %47 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 3]> : vector<2xi64>,   strides = dense<[1, 1]> : vector<2xi64>} ins(%45, %44 : tensor<1x2x5x9xbf16>, tensor<3x2x3x3xbf16>) outs(%46 : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
    return %47 : tensor<1x3x3x3xbf16>
  }
}
