module {
  func.func @main(%a: tensor<16x32x128xi8>) -> tensor<128xi8> {
    %c = arith.constant 0 : i8
    %e = tensor.empty() : tensor<128xi8>
    %f = linalg.fill ins(%c : i8) outs(%e : tensor<128xi8>) -> tensor<128xi8>
    %r = linalg.reduce ins(%a : tensor<16x32x128xi8>) outs(%f : tensor<128xi8>) dimensions = [0, 1]
      (%in: i8, %init: i8) {
        %s = arith.addi %in, %init : i8
        linalg.yield %s : i8
      }
    return %r : tensor<128xi8>
  }
}
