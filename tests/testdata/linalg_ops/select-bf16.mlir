module attributes {tf_saved_model.semantics} {
  func.func @main(%1470: tensor<1x2x128x1xi1>, %1468: tensor<1x2x128x1xbf16>, %1467: tensor<1x2x128x1xbf16>) -> tensor<1x2x128x1xbf16> {
    %1466 = "tensor.empty"() : () -> tensor<1x2x128x1xbf16>
    %1471 = "linalg.generic"(%1470, %1468, %1467, %1466) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%arg13057: i1, %arg13058: bf16, %arg13059: bf16, %arg13060: bf16):
      %30456 = "arith.select"(%arg13057, %arg13058, %arg13059) : (i1, bf16, bf16) -> bf16
      "linalg.yield"(%30456) : (bf16) -> ()
    }) : (tensor<1x2x128x1xi1>, tensor<1x2x128x1xbf16>, tensor<1x2x128x1xbf16>, tensor<1x2x128x1xbf16>) -> tensor<1x2x128x1xbf16>
    return %1471 : tensor<1x2x128x1xbf16>
  }
}
