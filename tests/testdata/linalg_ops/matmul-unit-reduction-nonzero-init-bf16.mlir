// Near miss for the unit-reduction matmul fold: the named matmul body
// accumulates into its init (addf(init, product)), so a k == 1 / n == 1 matmul
// over a *non-zero* init is not a bare product and must stay a matmul (the
// fold's zero-init guard declines it). bf16 is the dtype pair of the converted
// real-model site; with the guard declining, the matmul survives the pass and
// the case compiles and passes numerics against the harness's bare-product
// numpy reference (np.matmul) for this file class. (The f32 variant is not
// constructible here: a non-zero-init f32 matmul is unsupported by the torq
// backend, so it would fail to compile for a backend reason unrelated to the
// fold.)
module {
  func.func @main(%arg0: tensor<9x1xbf16>, %arg1: tensor<1x1xbf16>) -> (tensor<9x1xbf16>) {
    %cst = arith.constant 1.000000e+00 : bf16
    %0 = tensor.empty() : tensor<9x1xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<9x1xbf16>) -> tensor<9x1xbf16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<9x1xbf16>, tensor<1x1xbf16>) outs(%1 : tensor<9x1xbf16>) -> tensor<9x1xbf16>
    return %2 : tensor<9x1xbf16>
  }
}
