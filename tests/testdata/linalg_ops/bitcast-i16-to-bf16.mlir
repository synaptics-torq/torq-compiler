module {
  func.func @main(%arg0: tensor<1x1024x64xi16>) -> (tensor<1x1024x64xbf16>) {
    %55 = tensor.empty() : tensor<1x1024x64xbf16>
    %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1024x64xi16>) outs(%55 : tensor<1x1024x64xbf16>) {
    ^bb0(%in: i16, %out: bf16):
        %57 = arith.bitcast %in : i16 to bf16
        linalg.yield %57 : bf16
    } -> tensor<1x1024x64xbf16>
    return %56 : tensor<1x1024x64xbf16>
  }
}
