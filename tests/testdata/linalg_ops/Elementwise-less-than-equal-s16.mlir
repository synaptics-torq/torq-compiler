module {
  func.func @main(%arg0: tensor<1x4x2xi16>) -> tensor<1x4x2xi1> {
    // using one signed constant input to force signed comparison
    %cst = arith.constant dense<[[[-2, -1], [0,  1], [2, -3], [1,  0]]]> : tensor<1x4x2xi16>
    %empty = tensor.empty() : tensor<1x4x2xi1>
    %res = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
    iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst : tensor<1x4x2xi16>, tensor<1x4x2xi16>) outs(%empty : tensor<1x4x2xi1>) {
    ^bb0(%x: i16, %y: i16, %unused: i1):
      %cmp = arith.cmpi sle, %x, %y : i16
      linalg.yield %cmp : i1
    } -> tensor<1x4x2xi1>
    return %res : tensor<1x4x2xi1>
  }
}
