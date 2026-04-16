module {
  func.func @main(%arg0: tensor<1x4x16xbf16>) -> tensor<1x4x16xbf16> {
    %cst = arith.constant dense<1.500000e+00> : tensor<1x4x16xbf16>
    %0 = tensor.empty() : tensor<1x4x16xbf16>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst, %arg0 : tensor<1x4x16xbf16>, tensor<1x4x16xbf16>) outs(%0 : tensor<1x4x16xbf16>) {
    ^bb0(%lhs: bf16, %rhs: bf16, %out: bf16):
      %2 = arith.subf %lhs, %rhs : bf16
      linalg.yield %2 : bf16
    } -> tensor<1x4x16xbf16>
    return %1 : tensor<1x4x16xbf16>
  }
}
