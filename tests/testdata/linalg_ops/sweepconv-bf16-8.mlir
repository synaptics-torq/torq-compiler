module {
  func.func @main(%129 : tensor<1x2x17xbf16>, %128 : tensor<3x2x7xbf16>) -> tensor<1x3x3xbf16> {
    %130 = arith.constant dense<0.000000e+00> : tensor<1x3x3xbf16>
    %131 = linalg.conv_1d_ncw_fcw {dilations = dense<[2]> : vector<1xi64>,   strides = dense<[2]> : vector<1xi64>} ins(%129, %128 : tensor<1x2x17xbf16>, tensor<3x2x7xbf16>) outs(%130 : tensor<1x3x3xbf16>) -> tensor<1x3x3xbf16>
    return %131 : tensor<1x3x3xbf16>
  }
}
