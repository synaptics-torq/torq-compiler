module {
  func.func @main(%40: tensor<1x1024x64xbf16>, %cst_bool: tensor<i1>) -> tensor<1x1024x64xbf16> {
    %cst_0 = arith.constant -1.000000e+00 : bf16
    %cst_1 = arith.constant 1.000000e+00 : bf16    %41 = tensor.empty() : tensor<1x1024x64xbf16>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40 : tensor<1x1024x64xbf16>) outs(%41 : tensor<1x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %43 = arith.cmpf olt, %cst_1, %in : bf16
      %44 = arith.select %43, %cst_1, %in : bf16
      %45 = arith.cmpf ogt, %cst_0, %44 : bf16
      %46 = arith.select %45, %cst_0, %44 : bf16
      linalg.yield %46 : bf16
    } -> tensor<1x1024x64xbf16>
    return %42 : tensor<1x1024x64xbf16>
  }
}
