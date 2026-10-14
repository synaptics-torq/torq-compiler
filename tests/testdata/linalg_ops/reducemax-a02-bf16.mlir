module {
  func.func @main(%a: tensor<16x32x128xbf16>) -> tensor<32xbf16> {
    %c = arith.constant 0xFF80 : bf16
    %e = tensor.empty() : tensor<32xbf16>
    %f = linalg.fill ins(%c : bf16) outs(%e : tensor<32xbf16>) -> tensor<32xbf16>
    %r = linalg.reduce ins(%a : tensor<16x32x128xbf16>) outs(%f : tensor<32xbf16>) dimensions = [0, 2]
      (%in: bf16, %init: bf16) {
        %s = arith.maximumf %in, %init : bf16
        linalg.yield %s : bf16
      }
    return %r : tensor<32xbf16>
  }
}
