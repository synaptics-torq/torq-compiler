// bf16 relu6 clamp whose min/max bounds arrive as 0-D constant *tensor inputs*
// (block args) rather than inlined constants in the body. This is how torch
// lowers an ONNX Clip/ReLU6 with constant min/max tensors. ClampOpConversion
// (matchUnorderedClampConstInputs) must recognise this and lower it to a single
// torq_hl.act clamp so it stays on the NSS slice. Without that, the generic is
// left for the Host/CSS fallback, which test_linalg_ops forces off
// (--torq-disable-host --torq-disable-css), so this case fails loudly.
module {
  func.func @main(%in: tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16> {
    %lo = arith.constant dense<0.000000e+00> : tensor<bf16>
    %hi = arith.constant dense<6.000000e+00> : tensor<bf16>
    %e = tensor.empty() : tensor<1x32x128xbf16>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%in, %lo, %hi : tensor<1x32x128xbf16>, tensor<bf16>, tensor<bf16>) outs(%e : tensor<1x32x128xbf16>) {
    ^bb0(%a: bf16, %l: bf16, %h: bf16, %o: bf16):
      %c0 = arith.cmpf ult, %a, %l : bf16
      %s0 = arith.select %c0, %l, %a : bf16
      %c1 = arith.cmpf ugt, %s0, %h : bf16
      %s1 = arith.select %c1, %h, %s0 : bf16
      linalg.yield %s1 : bf16
    } -> tensor<1x32x128xbf16>
    return %r : tensor<1x32x128xbf16>
  }
}
