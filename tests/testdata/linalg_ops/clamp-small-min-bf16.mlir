// One-sided clamp with a small minimum (1e-7) given as a 0-d tensor operand, as torch-mlir
// lowers onnx.Clip(x, min). The bound must reach torq_hl.act unchanged: a decimal
// conversion with six digits used to turn it into 0.
//
// Compiled NSS-only (--torq-disable-host --torq-disable-css).
module {
  func.func @main(%arg0: tensor<1x576xbf16>) -> tensor<1x576xbf16> {
    %min = arith.constant dense<1.001170e-07> : tensor<bf16>
    %0 = tensor.empty() : tensor<1x576xbf16>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %min : tensor<1x576xbf16>, tensor<bf16>) outs(%0 : tensor<1x576xbf16>) {
    ^bb0(%in: bf16, %lo: bf16, %out: bf16):
      %c = arith.cmpf ult, %in, %lo : bf16
      %s = arith.select %c, %lo, %in : bf16
      linalg.yield %s : bf16
    } -> tensor<1x576xbf16>
    return %1 : tensor<1x576xbf16>
  }
}
