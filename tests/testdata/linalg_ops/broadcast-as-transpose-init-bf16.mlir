// Near miss for the key head broadcast rewrite: the broadcast is the *init* of
// the K^T-shaped transpose, not its data input, so it must not be moved.
#small = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%arg0: tensor<1x8x8xbf16>, %arg1: tensor<1x4x8x8xbf16>) -> tensor<1x4x8x8xbf16> {
    %init = tensor.empty() : tensor<1x4x8x8xbf16>
    %0 = linalg.generic {indexing_maps = [#small, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<1x8x8xbf16>) outs(%init : tensor<1x4x8x8xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4x8x8xbf16>
    %1 = linalg.transpose ins(%arg1 : tensor<1x4x8x8xbf16>) outs(%0 : tensor<1x4x8x8xbf16>) permutation = [0, 1, 3, 2]
    return %1 : tensor<1x4x8x8xbf16>
  }
}
