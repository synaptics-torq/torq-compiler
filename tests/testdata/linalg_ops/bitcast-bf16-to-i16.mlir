module {
  func.func @main(%arg0: tensor<1x1024x64xbf16>) -> (tensor<1x1024x64xi16>) {
  %3 = tensor.empty() : tensor<1x1024x64xi16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1024x64xbf16>) outs(%3 : tensor<1x1024x64xi16>) {
  ^bb0(%in: bf16, %out: i16):
    %57 = arith.bitcast %in : bf16 to i16
    linalg.yield %57 : i16
  } -> tensor<1x1024x64xi16>
    return %4 : tensor<1x1024x64xi16>
  }
}
