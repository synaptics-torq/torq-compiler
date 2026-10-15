module {
  func.func @main(%a: tensor<8x4xi16>) -> tensor<8xi16> {
    %c = arith.constant 20000 : i16
    %e = tensor.empty() : tensor<8xi16>
    %f = linalg.fill ins(%c : i16) outs(%e : tensor<8xi16>) -> tensor<8xi16>
    %r = linalg.reduce ins(%a : tensor<8x4xi16>) outs(%f : tensor<8xi16>) dimensions = [1]
      (%in: i16, %init: i16) {
        %s = arith.minui %in, %init : i16
        linalg.yield %s : i16
      }
    return %r : tensor<8xi16>
  }
}
