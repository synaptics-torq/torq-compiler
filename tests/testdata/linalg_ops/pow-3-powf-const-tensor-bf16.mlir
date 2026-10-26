// bf16 x^3 as math.powf whose exponent is a splat constant tensor in input 0. PowToMulPattern
// drops the exponent input, so x moves to input 0 of the mul(mul(x, x), x) expansion.
//
// Compiled NSS-only (--torq-disable-host --torq-disable-css).
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x196x24xbf16>) -> tensor<1x196x24xbf16> {
    %exp = arith.constant dense<3.000000e+00> : tensor<1x196x24xbf16>
    %0 = tensor.empty() : tensor<1x196x24xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%exp, %arg0 : tensor<1x196x24xbf16>, tensor<1x196x24xbf16>) outs(%0 : tensor<1x196x24xbf16>) {
    ^bb0(%e: bf16, %in: bf16, %out: bf16):
      %2 = math.powf %in, %e : bf16
      linalg.yield %2 : bf16
    } -> tensor<1x196x24xbf16>
    return %1 : tensor<1x196x24xbf16>
  }
}
