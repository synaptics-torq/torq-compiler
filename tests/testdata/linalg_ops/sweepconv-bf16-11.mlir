module {
  func.func @main(%160 : tensor<1x2x21xbf16>, %159 : tensor<3x2x8xbf16>) -> tensor<1x3x3xbf16> {
    %161 = arith.constant dense<0.000000e+00> : tensor<1x3x3xbf16>
    %162 = linalg.conv_1d_ncw_fcw {dilations = dense<[2]> : vector<1xi64>,   strides = dense<[3]> : vector<1xi64>} ins(%160, %159 : tensor<1x2x21xbf16>, tensor<3x2x8xbf16>) outs(%161 : tensor<1x3x3xbf16>) -> tensor<1x3x3xbf16>
    return %162 : tensor<1x3x3xbf16>
  }
}
