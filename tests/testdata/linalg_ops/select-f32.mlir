module attributes {tf_saved_model.semantics} {
  func.func @main(%1470: tensor<1x1x1x1024xi1>, %1468: tensor<1x1x1x1024xf32>, %1467: tensor<1x1x1x1024xf32>) -> tensor<1x1x1x1024xf32> {
    %1466 = "tensor.empty"() : () -> tensor<1x1x1x1024xf32>
    %1471 = "linalg.generic"(%1470, %1468, %1467, %1466) <{indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ], 
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>,
     #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> 
     ({
    ^bb0(%arg13057: i1, %arg13058: f32, %arg13059: f32, %arg13060: f32):
      %30456 = "arith.select"(%arg13057, %arg13058, %arg13059) : (i1, f32, f32) -> f32
      "linalg.yield"(%30456) : (f32) -> ()
    }) : (tensor<1x1x1x1024xi1>, tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) -> 
    tensor<1x1x1x1024xf32>
    return %1471 : tensor<1x1x1x1024xf32>
  }
}
