module {
  func.func @main(%190 : tensor<1x2x26xbf16>, %189 : tensor<3x2x8xbf16>) -> tensor<1x3x3xbf16> {
    %191 = arith.constant dense<0.000000e+00> : tensor<1x3x3xbf16>
    %192 = linalg.conv_1d_ncw_fcw {dilations = dense<[3]> : vector<1xi64>,   strides = dense<[2]> : vector<1xi64>} ins(%190, %189 : tensor<1x2x26xbf16>, tensor<3x2x8xbf16>) outs(%191 : tensor<1x3x3xbf16>) -> tensor<1x3x3xbf16>
    return %192 : tensor<1x3x3xbf16>
  }
}
