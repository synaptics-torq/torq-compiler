// Regression test: linalg.matmul whose constant RHS is addressed through a
// transposed indexing map ((d0, d1, d2) -> (d1, d2)), i.e. the weight is kept
// in [N, K] orientation (as produced by an ONNX Gemm with transB=1, e.g. the
// deit_tiny classifier). The torq_hl matmul lowering expects B in [K, N]
// orientation; forwarding the [N, K] operand unchanged used to crash with
// "matA.dim(MatA::K) == matB.dim(K)" in the TorqHW MatMul lowering.
module {
  func.func @main(%arg0: tensor<1x8xbf16>) -> (tensor<1x4xf32>) {
    %cst = arith.constant dense<[[0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.0, 3.0],
                                 [1.0, 0.125, -0.5, 2.0, 0.75, -1.5, 0.5, -3.0],
                                 [-0.25, 2.0, 1.0, -1.0, 0.5, 0.375, -0.875, 1.25],
                                 [3.0, -0.5, 0.25, 1.0, -2.0, 0.625, 1.75, -1.125]]> : tensor<4x8xbf16>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<1x4xf32>
    %0 = linalg.fill ins(%cst_0 : f32) outs(%init : tensor<1x4xf32>) -> tensor<1x4xf32>
    %1 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%arg0, %cst : tensor<1x8xbf16>, tensor<4x8xbf16>) outs(%0 : tensor<1x4xf32>) -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>
  }
}
