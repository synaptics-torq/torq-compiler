module {
  func.func @main(%0 : tensor<1x3x8200xbf16>, %1 : tensor<3x4096xbf16>) -> tensor<1x3x4xf32> {
    %2 = arith.constant dense<0.000000e+00> : tensor<1x3x4xf32>
    %3 = linalg.depthwise_conv_1d_ncw_cw {dilations = dense<[2]> : vector<1xi64>, strides = dense<[3]> : vector<1xi64>} ins(%0, %1 : tensor<1x3x8200xbf16>, tensor<3x4096xbf16>) outs(%2 : tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    return %3 : tensor<1x3x4xf32>
  }
}
