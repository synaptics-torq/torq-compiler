#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<8x96xi8>, %b: tensor<8x96xi8>) -> tensor<8x96xi8> {
    %e = tensor.empty() : tensor<8x96xi8>
    %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<8x96xi8>, tensor<8x96xi8>) outs(%e : tensor<8x96xi8>) {
    ^bb0(%x: i8, %y: i8, %o: i8):
      %s = arith.maxui %x, %y : i8
      linalg.yield %s : i8
    } -> tensor<8x96xi8>
    return %r : tensor<8x96xi8>
  }
}