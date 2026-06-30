// Native multi-channel Conv1D (C=16, F=32, Kw=7), no bias, NSS-only.
// Covers a wide kernel (Kw>3) on the native path.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(
      %arg0: tensor<1x16x48xbf16>,
      %arg1: tensor<32x16x7xbf16>
  ) -> tensor<1x32x42xbf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x32x42xf32>
    %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
        ins(%arg0, %arg1 : tensor<1x16x48xbf16>, tensor<32x16x7xbf16>)
        outs(%cst : tensor<1x32x42xf32>) -> tensor<1x32x42xf32>
    %1 = tensor.empty() : tensor<1x32x42xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%0 : tensor<1x32x42xf32>) outs(%1 : tensor<1x32x42xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %3 = arith.truncf %in : f32 to bf16
      linalg.yield %3 : bf16
    } -> tensor<1x32x42xbf16>
    return %2 : tensor<1x32x42xbf16>
  }
}
