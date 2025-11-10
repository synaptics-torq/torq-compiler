module {
  func.func @main(%arg0: tensor<1x1024x64xi16>, %arg2: tensor<1x1024x64xi16>) -> (tensor<1x1024x64xi1>) {
    %15 = tensor.empty() : tensor<1x1024x64xi1>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
    iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg2 : tensor<1x1024x64xi16>, tensor<1x1024x64xi16>) outs(%15 : tensor<1x1024x64xi1>) {
    ^bb0(%in: i16, %in_8: i16, %out: i1):
    %57 = arith.cmpi ult, %in, %in_8 : i16
    linalg.yield %57 : i1
  } -> tensor<1x1024x64xi1>
    return %16 : tensor<1x1024x64xi1>
  }
}
