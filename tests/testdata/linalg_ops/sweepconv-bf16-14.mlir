module {
  func.func @main(%207 : tensor<1x2x25xbf16>, %206 : tensor<3x2x7xbf16>) -> tensor<1x3x3xbf16> {
    %208 = arith.constant dense<0.000000e+00> : tensor<1x3x3xbf16>
    %209 = linalg.conv_1d_ncw_fcw {dilations = dense<[3]> : vector<1xi64>,   strides = dense<[3]> : vector<1xi64>} ins(%207, %206 : tensor<1x2x25xbf16>, tensor<3x2x7xbf16>) outs(%208 : tensor<1x3x3xbf16>) -> tensor<1x3x3xbf16>
    return %209 : tensor<1x3x3xbf16>
  }
}
