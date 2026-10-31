// Constant-weight dilated conv1d (dilation 2, k 3, expanded 5 in range).
// A named conv1d has no native weight-dilation path in the default pipeline
// (the conv1d->conv2d promotion refuses dilations and no pattern takes a
// named dilated conv1d), so this conv must take the residue split. If it
// bailed on the native check instead, it would fall to the CSS host, which
// --torq-disable-css turns into a compile error.
module {
  func.func @main(%0 : tensor<1x2x16xbf16>) -> tensor<1x4x12xf32> {
    %1 = arith.constant dense<0.01> : tensor<4x2x3xbf16>
    %2 = arith.constant dense<0.000000e+00> : tensor<1x4x12xf32>
    %3 = linalg.conv_1d_ncw_fcw {dilations = dense<[2]> : vector<1xi64>, strides = dense<[1]> : vector<1xi64>} ins(%0, %1 : tensor<1x2x16xbf16>, tensor<4x2x3xbf16>) outs(%2 : tensor<1x4x12xf32>) -> tensor<1x4x12xf32>
    return %3 : tensor<1x4x12xf32>
  }
}
