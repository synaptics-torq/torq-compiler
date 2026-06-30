// Batched (B>=2) MatMul with a constant weight: the weight broadcasts
// [K,N] -> [B,K,N]. Exercises the constant-weight batch-broadcast path:
//   - #1767: expandConstantInput must NOT rewrite the broadcast to an (invalid)
//     expand_shape (that aborted on a DenseElementsAttr::reshape assert).
//   - #1529 follow-up: the broadcast is absorbed into the batch_matmul via
//     indexing maps (no materialized [B,K,N] buffer), rather than left explicit.
// Non-uniform weights so a wrong broadcast axis/transpose is caught numerically.
module {
  func.func @main(%arg0: !torch.vtensor<[2,1,16],bf16>) -> !torch.vtensor<[2,1,8],bf16>
      attributes {torch.onnx_meta.opset_version = 22 : si64} {
    %0 = torch.operator "onnx.Constant"() {
           torch.onnx.value = dense<[
      [1.691, -0.466, 0.033, 0.408, -0.789, 0.002, -0.001, -1.755],
      [1.018, 0.600, -0.625, -0.172, 0.505, -0.261, -0.243, -1.453],
      [0.555, 0.124, 0.274, -1.527, 1.651, 0.154, -0.387, 2.029],
      [-0.045, -1.451, -0.405, -2.288, 1.049, -0.416, -0.743, 1.072],
      [-1.651, 0.535, -2.064, -0.662, -1.204, 1.462, 1.766, -0.329],
      [0.841, -0.180, 0.568, -0.753, -1.708, -1.803, 0.383, 2.248],
      [0.269, -0.525, 1.912, 0.237, 0.101, 0.253, -0.132, -0.309],
      [-1.435, 0.502, -0.095, 1.193, -0.369, -1.906, -0.100, 1.700],
      [-0.383, -0.890, -1.194, -1.050, -0.300, -1.180, 1.498, -0.283],
      [0.109, 1.438, 1.503, -0.213, 0.332, 0.735, -0.193, -1.778],
      [0.655, 0.894, 0.416, -0.924, -0.196, -0.591, -0.300, 1.297],
      [1.530, 0.669, 0.549, 0.677, -0.012, -0.076, -0.674, -0.056],
      [2.260, 0.869, -0.342, -0.472, -0.864, 0.374, 0.392, -1.443],
      [0.486, -0.569, 1.427, 0.157, 1.718, -0.458, -0.288, 0.300],
      [1.056, 0.566, -1.234, 0.183, 0.022, -0.429, -0.648, 1.748],
      [-0.390, -0.846, 0.637, 0.131, -0.076, 0.781, 0.489, 0.362]
    ]> : tensor<16x8xbf16>
         } : () -> !torch.vtensor<[16,8],bf16>
    %1 = torch.operator "onnx.MatMul"(%arg0, %0) :
           (!torch.vtensor<[2,1,16],bf16>, !torch.vtensor<[16,8],bf16>)
           -> !torch.vtensor<[2,1,8],bf16>
    return %1 : !torch.vtensor<[2,1,8],bf16>
  }
}
