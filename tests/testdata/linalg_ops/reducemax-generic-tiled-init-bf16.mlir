module {
  func.func @main(%a: tensor<1x131072xbf16>) -> tensor<1xbf16> {
    %c = arith.constant 100.0 : bf16
    %e = tensor.empty() : tensor<1xbf16>
    %f = linalg.fill ins(%c : bf16) outs(%e : tensor<1xbf16>) -> tensor<1xbf16>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%a : tensor<1x131072xbf16>) outs(%f : tensor<1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %m = arith.maximumf %in, %out : bf16
      linalg.yield %m : bf16
    } -> tensor<1xbf16>
    return %r : tensor<1xbf16>
  }
}
