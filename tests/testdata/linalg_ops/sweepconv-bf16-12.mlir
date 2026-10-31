module {
  func.func @main(%173 : tensor<1x2x23xbf16>, %172 : tensor<3x2x7xbf16>) -> tensor<1x3x3xbf16> {
    %174 = arith.constant dense<0.000000e+00> : tensor<1x3x3xbf16>
    %175 = linalg.conv_1d_ncw_fcw {dilations = dense<[3]> : vector<1xi64>,   strides = dense<[2]> : vector<1xi64>} ins(%173, %172 : tensor<1x2x23xbf16>, tensor<3x2x7xbf16>) outs(%174 : tensor<1x3x3xbf16>) -> tensor<1x3x3xbf16>
    return %175 : tensor<1x3x3xbf16>
  }
}
