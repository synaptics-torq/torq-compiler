module {
  func.func @main(%a: tensor<8x96xi8>) -> tensor<8xi8> {
    %c = arith.constant -128 : i8
    %e = tensor.empty() : tensor<8xi8>
    %f = linalg.fill ins(%c : i8) outs(%e : tensor<8xi8>) -> tensor<8xi8>
    %r = linalg.reduce ins(%a : tensor<8x96xi8>) outs(%f : tensor<8xi8>) dimensions = [1]
      (%in: i8, %init: i8) {
        %s = arith.minsi %in, %init : i8
        linalg.yield %s : i8
      }
    return %r : tensor<8xi8>
  }
}
