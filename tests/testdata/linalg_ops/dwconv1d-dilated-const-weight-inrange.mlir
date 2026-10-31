// Constant-weight dilated 1-D depthwise conv (dilation 2, k 10, expanded 19
// within the 1024 1-D depthwise cap).  Like conv1d, a named dwconv1d has no
// native weight-dilation path in the default pipeline, so this conv must
// take the residue split instead of falling to the CSS host.
module {
  func.func @main(%0 : tensor<1x8x64xbf16>) -> tensor<1x8x36xf32> {
    %1 = arith.constant dense<0.01> : tensor<8x10xbf16>
    %2 = arith.constant dense<0.000000e+00> : tensor<1x8x36xf32>
    %3 = linalg.depthwise_conv_1d_ncw_cw {dilations = dense<[2]> : vector<1xi64>, strides = dense<[1]> : vector<1xi64>} ins(%0, %1 : tensor<1x8x64xbf16>, tensor<8x10xbf16>) outs(%2 : tensor<1x8x36xf32>) -> tensor<1x8x36xf32>
    return %3 : tensor<1x8x36xf32>
  }
}
