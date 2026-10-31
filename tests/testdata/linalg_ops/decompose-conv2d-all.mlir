module {
  func.func @main(%0 : tensor<1x2x37x25xbf16>, %1 : tensor<4x2x15x10xbf16>) -> tensor<1x4x3x4xf32> {
    %2 = arith.constant dense<0.000000e+00> : tensor<1x4x3x4xf32>
    %3 = linalg.conv_2d_nchw_fchw {dilations = dense<[2, 2]> : vector<2xi64>, strides = dense<[4, 2]> : vector<2xi64>} ins(%0, %1 : tensor<1x2x37x25xbf16>, tensor<4x2x15x10xbf16>) outs(%2 : tensor<1x4x3x4xf32>) -> tensor<1x4x3x4xf32>
    return %3 : tensor<1x4x3x4xf32>
  }
}
