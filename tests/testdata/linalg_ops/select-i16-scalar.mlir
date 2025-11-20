// This test case triggers a corner case in cewr() generation when we have a single weigths that
// needs to be duplicated (fixed in PR#442)
module attributes {tf_saved_model.semantics} {
  func.func @main(%1470: tensor<i1>, %1468: tensor<i16>, %1467: tensor<i16>) -> tensor<i16> {
    %1466 = "tensor.empty"() : () -> tensor<i16>
    %1471 = "linalg.generic"(%1470, %1468, %1467, %1466) <{indexing_maps = [
      affine_map<() -> ()>,
      affine_map<() -> ()>,
      affine_map<() -> ()>,
      affine_map<() -> ()>
    ], 
    iterator_types = [], operandSegmentSizes = array<i32: 3, 1>}> 
     ({
    ^bb0(%arg13057: i1, %arg13058: i16, %arg13059: i16, %arg13060: i16):
      %30456 = "arith.select"(%arg13057, %arg13058, %arg13059) : (i1, i16, i16) -> i16
      "linalg.yield"(%30456) : (i16) -> ()
    }) : (tensor<i1>, tensor<i16>, tensor<i16>, tensor<i16>) -> 
    tensor<i16>
    return %1471 : tensor<i16>
  }
}
