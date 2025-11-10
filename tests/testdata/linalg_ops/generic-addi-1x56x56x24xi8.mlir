#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>

#attrs = {
  indexing_maps = [#map1, #map1, #map0],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
}

module {
  func.func @main(%arg0: tensor<1x56x56x24xi8>, %arg1: tensor<1x56x56x24xi8>) -> (tensor<1x56x56x24xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tensor.empty() : tensor<1x56x56x24xi8>
    %1 = linalg.generic #attrs
    ins(%arg0, %arg1 : tensor<1x56x56x24xi8>, tensor<1x56x56x24xi8>)
    outs(%0 : tensor<1x56x56x24xi8>) {
    ^bb0(%in0: i8, %in1: i8, %out: i8):
      %2 = arith.addi %in0, %in1 : i8
      linalg.yield %2 : i8
    } -> tensor<1x56x56x24xi8>
    return %1 : tensor<1x56x56x24xi8>
  }
}
