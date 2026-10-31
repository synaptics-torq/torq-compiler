// Regression for the H<->W transpose guard in the Conv2D fusion plan: when
// the conv's epilogue chain contains a transpose that swaps H and W in place
// ([0,1,3,2]), the plan must be cut back to before the transpose.  Otherwise
// the bias is folded into the torq_hl.conv2d while the bias-add generic
// survives to read the torq op's output and apply the bias a second time.
// TORQ_FP_MAX_TOL: 0.05
module {
  func.func @main(%0 : tensor<1x2x8x6xbf16>) -> tensor<1x4x4x6xbf16> {
    %1 = arith.constant dense<0.01> : tensor<4x2x3x3xbf16>
    %2 = arith.constant dense<0.000000e+00> : tensor<1x4x6x4xbf16>
    %3 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%0, %1 : tensor<1x2x8x6xbf16>, tensor<4x2x3x3xbf16>) outs(%2 : tensor<1x4x6x4xbf16>) -> tensor<1x4x6x4xbf16>
    %emptyT = tensor.empty() : tensor<1x4x4x6xbf16>
    %transposed = linalg.transpose ins(%3 : tensor<1x4x6x4xbf16>) outs(%emptyT : tensor<1x4x4x6xbf16>) permutation = [0, 1, 3, 2]
    %bias = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xbf16>
    %emptyB = tensor.empty() : tensor<1x4x4x6xbf16>
    %zero = arith.constant 0.000000e+00 : bf16
    %initB = linalg.fill ins(%zero : bf16) outs(%emptyB : tensor<1x4x4x6xbf16>) -> tensor<1x4x4x6xbf16>
    %bcast = linalg.broadcast ins(%bias : tensor<4xbf16>) outs(%initB : tensor<1x4x4x6xbf16>) dimensions = [0, 2, 3]
    %added = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed, %bcast : tensor<1x4x4x6xbf16>, tensor<1x4x4x6xbf16>) outs(%initB : tensor<1x4x4x6xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %o: bf16):
      %s = arith.addf %a, %b : bf16
      linalg.yield %s : bf16
    } -> tensor<1x4x4x6xbf16>
    return %added : tensor<1x4x4x6xbf16>
  }
}
