// Regression: large multi-channel Conv1D (C=128, W=2000), compiled NSS-only
// (--torq-disable-host --torq-disable-css by the linalg_ops harness).
//
// Historically this shape overflowed the im2col LRAM budget and the old
// conv1d->matmul lowering had to tile the unfold along the output width. That
// lowering is gone: the conv1d now promotes to conv2d (unit H extent, 7x1
// kernel, stride 1), which is within the hardware limits and lowers through
// the 2D conv path.
//
// Filter is a function argument (not an embedded constant) to keep the test small.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(
      %arg0: tensor<1x128x2000xbf16>, %arg1: tensor<64x128x7xbf16>
  ) -> tensor<1x64x1994xbf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x64x1994xf32>
    %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
        ins(%arg0, %arg1 : tensor<1x128x2000xbf16>, tensor<64x128x7xbf16>)
        outs(%cst : tensor<1x64x1994xf32>) -> tensor<1x64x1994xf32>
    %1 = tensor.empty() : tensor<1x64x1994xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%0 : tensor<1x64x1994xf32>) outs(%1 : tensor<1x64x1994xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %3 = arith.truncf %in : f32 to bf16
      linalg.yield %3 : bf16
    } -> tensor<1x64x1994xbf16>
    return %2 : tensor<1x64x1994xbf16>
  }
}
