// Regression: depthwise Conv1D (kernel 5, stride 2) with asymmetric SAME padding [1, 2].
//
// This is an effectively-1D conv (W=1, kernel 1, stride 1 along W). The asymmetric pad is
// materialized by tensor.pad into an odd-width (67) buffer, leaving a valid (zero-pad) depthwise
// conv over it. torq-convert-conv-valid-to-same used to make the odd height even by rewriting
// the conv into asymmetric hardware SAME padding (1, 2); the NPU mis-executes that for such
// 1D stride-2 convs, corrupting ~99% of the outputs. The conv must stay valid (odd dim grown to
// even by appending one zero at the high end). True 2D stride-2 convs still use SAME padding.
//
// Compiled NSS-only (--torq-disable-host --torq-disable-css) to force the conv onto the NPU path.
#map0 = affine_map<(d0) -> (d0)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%arg0: tensor<1x16x64xbf16>) -> tensor<1x16x32xbf16> {
    %pad_val = arith.constant 0.000000e+00 : bf16
    %filter = arith.constant dense<1.250000e-01> : tensor<16x5x1xbf16>
    // Non-splat so the extf+broadcast folds into the conv bias instead of a linalg.fill.
    %bias = arith.constant dense<[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
                                  0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6]> : tensor<16xbf16>

    %padded = tensor.pad %arg0 low[0, 0, 1] high[0, 0, 2] {
    ^bb0(%i0: index, %i1: index, %i2: index):
      tensor.yield %pad_val : bf16
    } : tensor<1x16x64xbf16> to tensor<1x16x67xbf16>
    %expanded = tensor.expand_shape %padded [[0], [1], [2, 3]] output_shape [1, 16, 67, 1]
        : tensor<1x16x67xbf16> into tensor<1x16x67x1xbf16>

    %bias_init = tensor.empty() : tensor<16xf32>
    %bias_f32 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
        ins(%bias : tensor<16xbf16>) outs(%bias_init : tensor<16xf32>) {
    ^bb0(%in: bf16, %o: f32):
      %e = arith.extf %in : bf16 to f32
      linalg.yield %e : f32
    } -> tensor<16xf32>
    %acc_init = tensor.empty() : tensor<1x16x32x1xf32>
    %acc = linalg.broadcast ins(%bias_f32 : tensor<16xf32>) outs(%acc_init : tensor<1x16x32x1xf32>)
        dimensions = [0, 2, 3]

    %conv = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<[2, 1]> : vector<2xi64>}
        ins(%expanded, %filter : tensor<1x16x67x1xbf16>, tensor<16x5x1xbf16>)
        outs(%acc : tensor<1x16x32x1xf32>) -> tensor<1x16x32x1xf32>

    %trunc_init = tensor.empty() : tensor<1x16x32x1xbf16>
    %trunc = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%conv : tensor<1x16x32x1xf32>) outs(%trunc_init : tensor<1x16x32x1xbf16>) {
    ^bb0(%in: f32, %o: bf16):
      %t = arith.truncf %in : f32 to bf16
      linalg.yield %t : bf16
    } -> tensor<1x16x32x1xbf16>
    %collapsed = tensor.collapse_shape %trunc [[0], [1], [2, 3]]
        : tensor<1x16x32x1xbf16> into tensor<1x16x32xbf16>
    return %collapsed : tensor<1x16x32xbf16>
  }
}
