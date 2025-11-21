module attributes {tf_saved_model.semantics} {
  func.func @main(%87: tensor<1x2x128x1xbf16>, %cast_1831: tensor<1x1x1x1xbf16>) -> (tensor<1x2x128x1xi1>) {
    %cst_78 = arith.constant 0.000000e+00 : f32
    %90 = tensor.empty() : tensor<1x2x128x1xi1>
    %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>,
     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
     ins(%87 : tensor<1x2x128x1xbf16>) outs(%90 : tensor<1x2x128x1xi1>) {
    ^bb0(%in: bf16, %out: i1):
      %1484 = arith.extf %in : bf16 to f32
      %1485 = arith.cmpf olt, %1484, %cst_78 : f32
      linalg.yield %1485 : i1
    } -> tensor<1x2x128x1xi1>
    return %91 : tensor<1x2x128x1xi1>
  }
}
