module {
  func.func @main(%0 : tensor<1x3x12x9xbf16>, %1 : tensor<3x3x3xbf16>) -> tensor<1x3x4x4xf32> {
    %2 = arith.constant dense<0.000000e+00> : tensor<1x3x4x4xf32>
    %3 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<[1, 1]> : vector<2xi64>, strides = dense<[3, 2]> : vector<2xi64>} ins(%0, %1 : tensor<1x3x12x9xbf16>, tensor<3x3x3xbf16>) outs(%2 : tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %3 : tensor<1x3x4x4xf32>
  }
}
