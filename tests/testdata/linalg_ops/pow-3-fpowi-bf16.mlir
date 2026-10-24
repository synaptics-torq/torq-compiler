// bf16 x^3 as math.fpowi with an i64 exponent (torch-mlir lowering of onnx.Pow), expanded by
// PowToMulPattern into mul(mul(x, x), x).
//
// Compiled NSS-only (--torq-disable-host --torq-disable-css).
module {
  func.func @main(%arg0: tensor<1x196x24xbf16>) -> tensor<1x196x24xbf16> {
    %c3_i64 = arith.constant 3 : i64
    %0 = tensor.empty() : tensor<1x196x24xbf16>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x196x24xbf16>) outs(%0 : tensor<1x196x24xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %2 = math.fpowi %in, %c3_i64 : bf16, i64
      linalg.yield %2 : bf16
    } -> tensor<1x196x24xbf16>
    return %1 : tensor<1x196x24xbf16>
  }
}
