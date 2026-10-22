#in = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @main(%a: tensor<16x32x128xi8>) -> tensor<16xi8> {
    %c = arith.constant 0 : i8
    %e = tensor.empty() : tensor<16xi8>
    %f = linalg.fill ins(%c : i8) outs(%e : tensor<16xi8>) -> tensor<16xi8>
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%a : tensor<16x32x128xi8>) outs(%f : tensor<16xi8>) {
      ^bb0(%in: i8, %o: i8):
        %s = arith.addi %in, %o : i8
        linalg.yield %s : i8
    } -> tensor<16xi8>
    return %r : tensor<16xi8>
  }
}
