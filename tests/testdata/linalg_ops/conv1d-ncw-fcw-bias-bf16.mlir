// Regression: native Conv1D per-channel bias under NSS-only compilation
// (--torq-disable-host --torq-disable-css). Extracted from the real NNNR3 model
// (NNNR3_0079_0.0960_torq_bf16, layer Conv_2): single-input-channel Conv1D,
// kernel width 3, stride 2, asymmetric pad [0,1], per-channel bias, bf16.
//
// The native conv1d path lowers to
//   conv1d(mul) -> reduce_sum(f32) -> collapse -> addf(bias) -> truncf(bf16)
// The fp32 bias add has no NSS/CSS lowering, so with Host disabled
// FoldConvBiasIntoReducePattern folds the bias into the reduce (applied in fp32
// by the activation stage), bit-exact with round_bf16(sum_f32 + bias_f32).
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x1x256xbf16>) -> tensor<1x32x128xbf16> {
      %cst = arith.constant dense<[5.031250e+00, 4.750000e+00, 4.031250e+00, 3.296880e+00, 3.609380e+00, 2.828130e+00, -2.796880e+00, 2.234380e+00, -4.656250e+00, -5.156250e+00, -4.437500e+00, -2.156250e+00, -3.875000e+00, 3.531250e+00, 2.937500e+00, -3.750000e+00, -3.515630e+00, 3.359380e+00, -4.406250e+00, 3.796880e+00, 3.515630e+00, -3.703130e+00, 1.976560e+00, 1.382810e+00, 3.890630e+00, -3.984380e+00, 3.968750e+00, 2.718750e+00, 3.390630e+00, -3.953130e+00, 1.351560e+00, -4.812500e+00]> : tensor<32xbf16>
      %cst_0 = arith.constant dense<[[[2.451170e-01, 4.472660e-01, -6.298830e-02]], [[3.339840e-01, 3.300780e-01, 1.794430e-02]], [[-1.826170e-01, 4.433590e-01, 3.847660e-01]], [[2.319340e-02, 1.020510e-01, 2.578130e-01]], [[-4.931640e-02, 3.027340e-01, 2.138670e-01]], [[1.298830e-01, -4.726560e-01, 7.539060e-01]], [[-4.941410e-01, -4.746090e-01, 4.492190e-01]], [[-1.035160e-01, -3.261720e-01, 7.968750e-01]], [[3.574220e-01, -5.156250e-01, -5.312500e-01]], [[3.535160e-01, -4.375000e-01, -7.304680e-01]], [[-9.082030e-02, -3.574220e-01, -2.412110e-01]], [[-2.636720e-02, 4.218750e-01, -7.031250e-01]], [[-5.468750e-02, -2.431640e-01, -2.373050e-01]], [[1.503910e-01, 2.089840e-01, 1.113280e-01]], [[1.195310e+00, -3.906250e-01, -4.023440e-01]], [[-9.414060e-01, -1.669920e-01, 5.625000e-01]], [[-9.687500e-01, 9.609370e-01, -5.156250e-01]], [[2.265630e-01, 1.230470e-01, 6.738280e-02]], [[-5.078130e-01, -2.080080e-01, 4.833980e-02]], [[3.945310e-01, -2.392580e-01, 3.417970e-01]], [[3.002930e-02, 3.144530e-01, 1.318360e-01]], [[-1.647950e-02, -5.039060e-01, 5.432130e-03]], [[7.109380e-01, -1.023440e+00, 5.273440e-01]], [[8.750000e-01, -8.320310e-01, 1.069340e-01]], [[3.247070e-02, 3.574220e-01, 1.191410e-01]], [[-2.656250e-01, -3.242190e-01, 5.322270e-02]], [[8.740230e-02, -1.787110e-01, 6.328130e-01]], [[-5.703130e-01, 8.593750e-01, 8.105460e-02]], [[4.238280e-01, -1.660160e-02, -1.635740e-02]], [[-2.373050e-01, -3.574220e-01, -5.920410e-03]], [[-4.199220e-01, 1.148440e+00, -5.390630e-01]], [[-2.343750e-01, -2.451170e-01, -2.109380e-01]]]> : tensor<32x1x3xbf16>
      %cst_1 = arith.constant 0.000000e+00 : bf16
      %padded = tensor.pad %arg0 low[0, 0, 0] high[0, 0, 1] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index):
        tensor.yield %cst_1 : bf16
      } : tensor<1x1x256xbf16> to tensor<1x1x257xbf16>
      %1 = tensor.empty() : tensor<1x32x128xf32>
      %2 = tensor.empty() : tensor<32xf32>
      %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<32xbf16>) outs(%2 : tensor<32xf32>) {
      ^bb0(%in: bf16, %out: f32):
        %9 = arith.extf %in : bf16 to f32
        linalg.yield %9 : f32
      } -> tensor<32xf32>
      %broadcasted = linalg.broadcast ins(%3 : tensor<32xf32>) outs(%1 : tensor<1x32x128xf32>) dimensions = [0, 2] 
      %4 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%padded, %cst_0 : tensor<1x1x257xbf16>, tensor<32x1x3xbf16>) outs(%broadcasted : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
      %5 = tensor.empty() : tensor<1x32x128xbf16>
      %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<1x32x128xf32>) outs(%5 : tensor<1x32x128xbf16>) {
      ^bb0(%in: f32, %out: bf16):
        %9 = arith.truncf %in : f32 to bf16
        linalg.yield %9 : bf16
      } -> tensor<1x32x128xbf16>
    return %6 : tensor<1x32x128xbf16>
  }
}
