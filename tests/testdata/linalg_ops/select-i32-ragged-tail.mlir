module attributes {tf_saved_model.semantics} {
  func.func @main(%c: tensor<1x300xi1>, %a: tensor<1x300xi32>, %b: tensor<1x300xi32>) -> tensor<1x300xi32> {
    %e = "tensor.empty"() : () -> tensor<1x300xi32>
    %r = "linalg.generic"(%c, %a, %b, %e) <{indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%x: i1, %y: i32, %z: i32, %w: i32):
      %s = "arith.select"(%x, %y, %z) : (i1, i32, i32) -> i32
      "linalg.yield"(%s) : (i32) -> ()
    }) : (tensor<1x300xi1>, tensor<1x300xi32>, tensor<1x300xi32>, tensor<1x300xi32>) -> tensor<1x300xi32>
    return %r : tensor<1x300xi32>
  }
}
