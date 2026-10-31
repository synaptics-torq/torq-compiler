module {
  func.func @main(%0 : tensor<1x2x12xbf16>, %1 : tensor<4x2x3xbf16>) -> tensor<1x4x4xf32> {
    %2 = arith.constant dense<0.000000e+00> : tensor<1x4x4xf32>
    %3 = linalg.conv_1d_ncw_fcw {dilations = dense<[1]> : vector<1xi64>, strides = dense<[3]> : vector<1xi64>} ins(%0, %1 : tensor<1x2x12xbf16>, tensor<4x2x3xbf16>) outs(%2 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    return %3 : tensor<1x4x4xf32>
  }
}
