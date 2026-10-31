// TORQ_FP_MAX_TOL: 0.05
module {
  func.func @main(%0 : tensor<1x2x40xbf16>, %1 : tensor<3x2x12xbf16>) -> tensor<1x3x8xbf16> {
    %2 = arith.constant dense<0.000000e+00> : tensor<1x3x8xbf16>
    %3 = linalg.conv_1d_ncw_fcw {dilations = dense<[1]> : vector<1xi64>, strides = dense<[4]> : vector<1xi64>} ins(%0, %1 : tensor<1x2x40xbf16>, tensor<3x2x12xbf16>) outs(%2 : tensor<1x3x8xbf16>) -> tensor<1x3x8xbf16>
    return %3 : tensor<1x3x8xbf16>
  }
}
