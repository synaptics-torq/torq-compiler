module  {
  func.func @main(%14: tensor<1x21x1024xbf16>) -> tensor<1x21x1024xbf16> attributes {} {
    %cst = arith.constant 1.500000e+00 : bf16
    %15 = tensor.empty() : tensor<1x21x1024xbf16>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<1x21x1024xbf16>) outs(%15 : tensor<1x21x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %18 = arith.subf %cst, %in : bf16
      linalg.yield %18 : bf16
    } -> tensor<1x21x1024xbf16>
    return %16 : tensor<1x21x1024xbf16>
  }
}
