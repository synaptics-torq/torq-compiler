// A stride-3 conv1d with 3 output points and a per-channel bias. The stride does not divide
// the kernel, so Conv1DtoMatmulPattern gathers the 3 input windows into a [3, C*K] matrix and
// lowers the conv to a matmul; the bias add and the truncf fold into the fully_connected op.
// TORQ_FP_MAX_TOL: 0.05
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
module {
  func.func @main(%0 : tensor<1x64x13xbf16>) -> tensor<1x128x3xbf16> {
    %w = arith.constant dense<0.01> : tensor<128x64x7xbf16>
    %b = arith.constant dense<0.5> : tensor<128xf32>
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<1x128x3xf32>
    %i = linalg.fill ins(%z : f32) outs(%e : tensor<1x128x3xf32>) -> tensor<1x128x3xf32>
    %c = linalg.conv_1d_ncw_fcw {dilations = dense<[1]> : vector<1xi64>, strides = dense<[3]> : vector<1xi64>} ins(%0, %w : tensor<1x64x13xbf16>, tensor<128x64x7xbf16>) outs(%i : tensor<1x128x3xf32>) -> tensor<1x128x3xf32>
    %a = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%c, %b : tensor<1x128x3xf32>, tensor<128xf32>) outs(%e : tensor<1x128x3xf32>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %v = arith.addf %in, %bias : f32
      linalg.yield %v : f32
    } -> tensor<1x128x3xf32>
    %e2 = tensor.empty() : tensor<1x128x3xbf16>
    %t = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%a : tensor<1x128x3xf32>) outs(%e2 : tensor<1x128x3xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %v = arith.truncf %in : f32 to bf16
      linalg.yield %v : bf16
    } -> tensor<1x128x3xbf16>
    return %t : tensor<1x128x3xbf16>
  }
}
