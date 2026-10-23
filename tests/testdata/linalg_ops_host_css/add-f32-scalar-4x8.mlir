#map0 = affine_map<(d0, d1) -> (d0, d1)>

#attrs = {
  indexing_maps = [#map0, #map0],
  iterator_types = ["parallel", "parallel"]
}

module {
  func.func @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
    %cst = arith.constant 1.5 : f32
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.generic #attrs
    ins(%arg0 : tensor<4x8xf32>)
    outs(%0 : tensor<4x8xf32>) {
    ^bb0(%in0: f32, %out: f32):
      %2 = arith.addf %in0, %cst : f32
      linalg.yield %2 : f32
    } -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }
}
