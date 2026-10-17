// Standalone bf16 scalar multiply where the scalar arrives as a captured f32
// constant width-cast in the body (arith.truncf) -- the importer's form for
// f32 scalars over bf16 data. MulOpPattern must fold the cast at pattern
// time (traceToConstFloat) and lower to a torq_hl.mul with the bf16 scalar
// as the WRAM weight. Previously the body's mulf was not recognized as an
// elementwise binary op with a constant operand and the generic had no
// lowering in NSS-only mode.
module {
  func.func public @main(%arg0: tensor<1x1024x64xbf16>) -> tensor<1x1024x64xbf16> {
    %cst = arith.constant 0.166666701 : f32
    %e = tensor.empty() : tensor<1x1024x64xbf16>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1024x64xbf16>) outs(%e : tensor<1x1024x64xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      %a = arith.truncf %cst : f32 to bf16
      %m = arith.mulf %in, %a : bf16
      linalg.yield %m : bf16
    } -> tensor<1x1024x64xbf16>
    return %r : tensor<1x1024x64xbf16>
  }
}
