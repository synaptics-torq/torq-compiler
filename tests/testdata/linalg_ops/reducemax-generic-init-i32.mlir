#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<8x96xi32>) -> tensor<8xi32> {
    %c = arith.constant 1073741824 : i32
    %e = tensor.empty() : tensor<8xi32>
    %f = linalg.fill ins(%c : i32) outs(%e : tensor<8xi32>) -> tensor<8xi32>
    %r = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%a : tensor<8x96xi32>) outs(%f : tensor<8xi32>) {
    ^bb0(%in: i32, %o: i32):
      %s = arith.maxsi %in, %o : i32
      linalg.yield %s : i32
    } -> tensor<8xi32>
    return %r : tensor<8xi32>
  }
}
