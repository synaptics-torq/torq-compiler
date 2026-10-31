module {
  func.func @main(%216 : tensor<1x2x28xbf16>, %215 : tensor<3x2x8xbf16>) -> tensor<1x3x3xbf16> {
    %217 = arith.constant dense<0.000000e+00> : tensor<1x3x3xbf16>
    %218 = linalg.conv_1d_ncw_fcw {dilations = dense<[3]> : vector<1xi64>,   strides = dense<[3]> : vector<1xi64>} ins(%216, %215 : tensor<1x2x28xbf16>, tensor<3x2x8xbf16>) outs(%217 : tensor<1x3x3xbf16>) -> tensor<1x3x3xbf16>
    return %218 : tensor<1x3x3xbf16>
  }
}
