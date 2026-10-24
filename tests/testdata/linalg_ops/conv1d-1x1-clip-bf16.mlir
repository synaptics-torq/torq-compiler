// 1x1 conv1d followed by truncf and a two-sided clip with 0-d tensor bounds (1e-8, 1e8), as in
// the filter bank of an ONNX audio front end. FuseReluClampWithTruncf merges the clip with the
// truncf; the conv1d to conv2d rewrite must keep that fused clip, not replace it with a plain
// truncf.
//
// Compiled NSS-only (--torq-disable-host --torq-disable-css).
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar = affine_map<(d0, d1, d2) -> ()>
module {
  func.func @main(%arg0: tensor<1x16x64xbf16>) -> tensor<1x8x64xbf16> {
    %zero = arith.constant 0.000000e+00 : f32
    %w = arith.constant dense<[[[0.5], [-0.25], [0.75], [0.125], [-0.5], [1.0], [0.25], [0.25], [1.5], [-1.0], [-0.125], [0.5], [0.375], [-0.75], [1.0], [2.0]],
                               [[-0.5], [0.25], [0.5], [-0.125], [0.5], [-1.0], [0.75], [0.25], [-1.5], [1.0], [0.125], [-0.5], [0.25], [0.75], [-1.0], [0.5]],
                               [[0.25], [0.25], [-0.75], [0.5], [1.0], [0.5], [-0.25], [0.125], [0.5], [-0.5], [0.25], [1.0], [-0.375], [0.5], [0.25], [-1.0]],
                               [[1.0], [-1.0], [0.5], [0.5], [-0.25], [0.25], [0.125], [-0.5], [0.75], [0.25], [-0.5], [0.5], [1.0], [-0.25], [0.5], [0.25]],
                               [[-0.25], [0.5], [1.0], [-0.5], [0.25], [0.75], [-1.0], [0.5], [0.125], [0.5], [1.0], [-0.25], [0.5], [0.25], [-0.75], [1.0]],
                               [[0.5], [0.5], [-0.5], [1.0], [0.25], [-0.25], [0.5], [0.75], [-1.0], [0.125], [0.5], [0.25], [-0.5], [1.0], [0.5], [-0.125]],
                               [[0.75], [-0.5], [0.25], [0.25], [1.0], [-1.0], [0.5], [0.5], [0.25], [-0.25], [0.75], [0.5], [0.125], [-0.5], [1.0], [0.25]],
                               [[-1.0], [0.25], [0.5], [0.75], [-0.5], [0.5], [0.25], [-0.25], [1.0], [0.5], [-0.125], [0.25], [0.5], [0.75], [-0.5], [0.5]]]> : tensor<8x16x1xbf16>
    %lo = arith.constant dense<1.001170e-08> : tensor<bf16>
    %hi = arith.constant dense<1.001390e+08> : tensor<bf16>
    %e0 = tensor.empty() : tensor<1x8x64xf32>
    %acc = linalg.fill ins(%zero : f32) outs(%e0 : tensor<1x8x64xf32>) -> tensor<1x8x64xf32>
    %conv = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
        ins(%arg0, %w : tensor<1x16x64xbf16>, tensor<8x16x1xbf16>) outs(%acc : tensor<1x8x64xf32>) -> tensor<1x8x64xf32>
    %e1 = tensor.empty() : tensor<1x8x64xbf16>
    %t = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%conv : tensor<1x8x64xf32>) outs(%e1 : tensor<1x8x64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %v = arith.truncf %in : f32 to bf16
      linalg.yield %v : bf16
    } -> tensor<1x8x64xbf16>
    %e2 = tensor.empty() : tensor<1x8x64xbf16>
    %c = linalg.generic {indexing_maps = [#map, #scalar, #scalar, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%t, %lo, %hi : tensor<1x8x64xbf16>, tensor<bf16>, tensor<bf16>) outs(%e2 : tensor<1x8x64xbf16>) {
    ^bb0(%in: bf16, %l: bf16, %h: bf16, %out: bf16):
      %c0 = arith.cmpf ult, %in, %l : bf16
      %s0 = arith.select %c0, %l, %in : bf16
      %c1 = arith.cmpf ugt, %s0, %h : bf16
      %s1 = arith.select %c1, %h, %s0 : bf16
      linalg.yield %s1 : bf16
    } -> tensor<1x8x64xbf16>
    return %c : tensor<1x8x64xbf16>
  }
}
