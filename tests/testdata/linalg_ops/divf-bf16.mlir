module attributes {tf_saved_model.semantics} {
  func.func @main(%cast_1810: tensor<1x1x1x64xbf16>, %cast_1831: tensor<1x1x1x1xbf16>) -> (tensor<1x1x1x64xbf16>) {
    %3432 = tensor.empty() : tensor<1x1x1x64xbf16>
    %3433 = linalg.generic {indexing_maps = 
    [affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cast_1810, %cast_1831 : tensor<1x1x1x64xbf16>, tensor<1x1x1x1xbf16>) 
    outs(%3432 : tensor<1x1x1x64xbf16>) {
    ^bb0(%in: bf16, %in_193267: bf16, %out: bf16):
      %182756 = arith.divf %in, %in_193267 : bf16
      linalg.yield %182756 : bf16
    } -> tensor<1x1x1x64xbf16>
    return %3433 : tensor<1x1x1x64xbf16>
  }
}
