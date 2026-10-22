#in = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (0, 0, d2)>
module {
  func.func @main(%a: tensor<16x32x128xi32>) -> tensor<1x1x128xi32> {
    %c = arith.constant 0 : i32
    %e = tensor.empty() : tensor<1x1x128xi32>
    %f = linalg.fill ins(%c : i32) outs(%e : tensor<1x1x128xi32>) -> tensor<1x1x128xi32>
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["reduction", "reduction", "parallel"]}
      ins(%a : tensor<16x32x128xi32>) outs(%f : tensor<1x1x128xi32>) {
      ^bb0(%in: i32, %o: i32):
        %s = arith.addi %in, %o : i32
        linalg.yield %s : i32
    } -> tensor<1x1x128xi32>
    return %r : tensor<1x1x128xi32>
  }
}
