module {
  func.func @main(%10: tensor<1x8x1962xi16>) -> (tensor<1x8x1962xi8>) {
    %15 = tensor.empty() : tensor<1x8x1962xi8>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%10 : tensor<1x8x1962xi16>) outs(%15 : tensor<1x8x1962xi8>) {
    ^bb0(%in: i16, %out: i8):
      %114 = arith.trunci %in : i16 to i8
      linalg.yield %114 : i8
    } -> tensor<1x8x1962xi8>
    return %16 : tensor<1x8x1962xi8>
  }
}
