// f32 -> bf16 truncf followed by a bf16 relu6 clamp. OptimizeLinalgForTorqPass fuses
// the two generics into a single "clamp in f32, then truncf to bf16" generic, which
// ClampOpConversion (matchFusedClampTruncf) lowers to a torq_hl.act that clamps in
// f32 and emits bf16 - keeping the op on the NSS slice (no Host/CSS fallback).
module {
  func.func @main(%in: tensor<1x1024x64xf32>) -> tensor<1x1024x64xbf16> {
    %low = arith.constant 0.000000e+00 : bf16
    %high = arith.constant 6.000000e+00 : bf16
    %e0 = tensor.empty() : tensor<1x1024x64xbf16>
    %t = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%in : tensor<1x1024x64xf32>) outs(%e0 : tensor<1x1024x64xbf16>) {
    ^bb0(%a: f32, %o: bf16):
      %tr = arith.truncf %a : f32 to bf16
      linalg.yield %tr : bf16
    } -> tensor<1x1024x64xbf16>
    %e1 = tensor.empty() : tensor<1x1024x64xbf16>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%t : tensor<1x1024x64xbf16>) outs(%e1 : tensor<1x1024x64xbf16>) {
    ^bb0(%a: bf16, %o: bf16):
      %c0 = arith.cmpf ult, %a, %low : bf16
      %s0 = arith.select %c0, %low, %a : bf16
      %c1 = arith.cmpf ugt, %s0, %high : bf16
      %s1 = arith.select %c1, %high, %s0 : bf16
      linalg.yield %s1 : bf16
    } -> tensor<1x1024x64xbf16>
    return %r : tensor<1x1024x64xbf16>
  }
}
