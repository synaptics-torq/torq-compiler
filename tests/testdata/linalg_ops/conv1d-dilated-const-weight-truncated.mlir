// Constant-weight dilated conv1d with a TRUNCATED output (8 positions
// instead of the 12 the full valid window allows).  Regression for the
// residue split's insert_slice running past the actual output extent: the
// dilation rewrite sizes each residue class from the thinned input, which
// over-runs a truncated output and used to fail with
// "slice along dimension 2 runs out-of-bounds".
module {
  func.func @main(%0 : tensor<1x2x16xbf16>) -> tensor<1x4x8xf32> {
    %1 = arith.constant dense<0.01> : tensor<4x2x3xbf16>
    %2 = arith.constant dense<0.000000e+00> : tensor<1x4x8xf32>
    %3 = linalg.conv_1d_ncw_fcw {dilations = dense<[2]> : vector<1xi64>, strides = dense<[1]> : vector<1xi64>} ins(%0, %1 : tensor<1x2x16xbf16>, tensor<4x2x3xbf16>) outs(%2 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    return %3 : tensor<1x4x8xf32>
  }
}
