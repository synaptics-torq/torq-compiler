module {
  func.func @main(%138 : tensor<1x2x19xbf16>, %137 : tensor<3x2x8xbf16>) -> tensor<1x3x3xbf16> {
    %139 = arith.constant dense<0.000000e+00> : tensor<1x3x3xbf16>
    %140 = linalg.conv_1d_ncw_fcw {dilations = dense<[2]> : vector<1xi64>,   strides = dense<[2]> : vector<1xi64>} ins(%138, %137 : tensor<1x2x19xbf16>, tensor<3x2x8xbf16>) outs(%139 : tensor<1x3x3xbf16>) -> tensor<1x3x3xbf16>
    return %140 : tensor<1x3x3xbf16>
  }
}
