// Native multi-channel Conv1D (C=48, F=48, Kw=3), NSS-only. Covers bias-fold-into-reduce with a
// runtime-arg filter, so the fp32 bias add never materializes (the im2col path forces a host
// fallback here; C=1 case is conv1d-ncw-fcw-bias-bf16.mlir).
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(
      %arg0: tensor<1x48x35xbf16>,
      %arg1: tensor<48x48x3xbf16>
  ) -> tensor<1x48x33xbf16> {
    %cst = arith.constant dense<[-1.210940e+00, -4.980470e-02, -1.406250e+00, 1.257810e+00, 5.546880e-01, -8.906250e-01, 1.796880e+00, 3.164060e-01, 4.687500e-01, 9.062500e-01, 2.910160e-01, 1.464840e-01, 2.099610e-01, -3.359380e-01, 5.937500e-01, -3.222660e-02, 4.902340e-01, 1.421880e+00, 1.464840e-01, -6.738280e-02, -2.597660e-01, 3.759770e-02, 4.023440e-01, 6.689450e-02, 4.414060e-01, 1.796880e-01, 9.335930e-01, -3.417970e-01, 1.250000e-01, -2.519530e-01, -1.748050e-01, 1.464840e-01, 5.117190e-01, -3.984380e-01, -6.640630e-01, 5.390630e-01, 1.343750e+00, 9.687500e-01, 4.570310e-01, -1.201170e-01, 5.273440e-01, 4.179690e-01, -1.156250e+00, 2.207030e-01, 3.281250e-01, 2.412110e-01, -3.281250e-01, -4.833980e-02]> : tensor<48xbf16>
    %1 = tensor.empty() : tensor<1x48x33xf32>
    %2 = tensor.empty() : tensor<48xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<48xbf16>) outs(%2 : tensor<48xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %9 = arith.extf %in : bf16 to f32
      linalg.yield %9 : f32
    } -> tensor<48xf32>
    %broadcasted = linalg.broadcast ins(%3 : tensor<48xf32>) outs(%1 : tensor<1x48x33xf32>) dimensions = [0, 2]
    %4 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
        ins(%arg0, %arg1 : tensor<1x48x35xbf16>, tensor<48x48x3xbf16>)
        outs(%broadcasted : tensor<1x48x33xf32>) -> tensor<1x48x33xf32>
    %5 = tensor.empty() : tensor<1x48x33xbf16>
    %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<1x48x33xf32>) outs(%5 : tensor<1x48x33xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %9 = arith.truncf %in : f32 to bf16
      linalg.yield %9 : bf16
    } -> tensor<1x48x33xbf16>
    return %6 : tensor<1x48x33xbf16>
  }
}
