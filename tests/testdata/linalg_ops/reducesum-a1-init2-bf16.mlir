module {
  func.func @main(%a: tensor<8x8xbf16>) -> tensor<8xbf16> {
    %c = arith.constant 2.0 : bf16
    %e = tensor.empty() : tensor<8xbf16>
    %f = linalg.fill ins(%c : bf16) outs(%e : tensor<8xbf16>) -> tensor<8xbf16>
    %r = linalg.reduce ins(%a : tensor<8x8xbf16>) outs(%f : tensor<8xbf16>) dimensions = [1]
      (%in: bf16, %init: bf16) {
        %s = arith.addf %in, %init : bf16
        linalg.yield %s : bf16
      }
    return %r : tensor<8xbf16>
  }
}
