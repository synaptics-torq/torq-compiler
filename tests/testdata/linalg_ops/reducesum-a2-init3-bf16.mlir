module {
  func.func @main(%a: tensor<4x8x16xbf16>) -> tensor<4x8xbf16> {
    %c = arith.constant 3.0 : bf16
    %e = tensor.empty() : tensor<4x8xbf16>
    %f = linalg.fill ins(%c : bf16) outs(%e : tensor<4x8xbf16>) -> tensor<4x8xbf16>
    %r = linalg.reduce ins(%a : tensor<4x8x16xbf16>) outs(%f : tensor<4x8xbf16>) dimensions = [2]
      (%in: bf16, %init: bf16) {
        %s = arith.addf %in, %init : bf16
        linalg.yield %s : bf16
      }
    return %r : tensor<4x8xbf16>
  }
}
