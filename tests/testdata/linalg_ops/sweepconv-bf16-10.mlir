module {
  func.func @main(%147 : tensor<1x2x19xbf16>, %146 : tensor<3x2x7xbf16>) -> tensor<1x3x3xbf16> {
    %148 = arith.constant dense<0.000000e+00> : tensor<1x3x3xbf16>
    %149 = linalg.conv_1d_ncw_fcw {dilations = dense<[2]> : vector<1xi64>,   strides = dense<[3]> : vector<1xi64>} ins(%147, %146 : tensor<1x2x19xbf16>, tensor<3x2x7xbf16>) outs(%148 : tensor<1x3x3xbf16>) -> tensor<1x3x3xbf16>
    return %149 : tensor<1x3x3xbf16>
  }
}
