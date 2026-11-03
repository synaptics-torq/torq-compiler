module attributes {tf_saved_model.semantics} {
  func.func @main(%c: tensor<1x300xi1>, %a: tensor<1x300xi16>, %b: tensor<1x300xi16>) -> tensor<1x300xi16> {
    %e = "tensor.empty"() : () -> tensor<1x300xi16>
    %r = "linalg.generic"(%c, %a, %b, %e) <{indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%x: i1, %y: i16, %z: i16, %w: i16):
      %s = "arith.select"(%x, %y, %z) : (i1, i16, i16) -> i16
      "linalg.yield"(%s) : (i16) -> ()
    }) : (tensor<1x300xi1>, tensor<1x300xi16>, tensor<1x300xi16>, tensor<1x300xi16>) -> tensor<1x300xi16>
    return %r : tensor<1x300xi16>
  }
}
