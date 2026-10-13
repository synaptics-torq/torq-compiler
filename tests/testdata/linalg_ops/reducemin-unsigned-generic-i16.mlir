#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<8x96xi16>) -> tensor<8xi16> {
    %c = arith.constant -1 : i16
    %e = tensor.empty() : tensor<8xi16>
    %f = linalg.fill ins(%c : i16) outs(%e : tensor<8xi16>) -> tensor<8xi16>
    %r = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%a : tensor<8x96xi16>) outs(%f : tensor<8xi16>) {
    ^bb0(%in: i16, %o: i16):
      %s = arith.minui %in, %o : i16
      linalg.yield %s : i16
    } -> tensor<8xi16>
    return %r : tensor<8xi16>
  }
}