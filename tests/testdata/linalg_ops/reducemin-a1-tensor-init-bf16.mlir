module {
  func.func @main(%a: tensor<8x4xbf16>, %b: tensor<8xbf16>) -> tensor<8xbf16> {
    %r = linalg.reduce ins(%a : tensor<8x4xbf16>) outs(%b : tensor<8xbf16>) dimensions = [1]
      (%in: bf16, %init: bf16) {
        %s = arith.minimumf %in, %init : bf16
        linalg.yield %s : bf16
      }
    return %r : tensor<8xbf16>
  }
}
