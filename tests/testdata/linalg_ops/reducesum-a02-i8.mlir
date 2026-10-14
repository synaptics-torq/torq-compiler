module {
  func.func @main(%a: tensor<16x32x128xi8>) -> tensor<32xi8> {
    %c = arith.constant 0 : i8
    %e = tensor.empty() : tensor<32xi8>
    %f = linalg.fill ins(%c : i8) outs(%e : tensor<32xi8>) -> tensor<32xi8>
    %r = linalg.reduce ins(%a : tensor<16x32x128xi8>) outs(%f : tensor<32xi8>) dimensions = [0, 2]
      (%in: i8, %init: i8) {
        %s = arith.addi %in, %init : i8
        linalg.yield %s : i8
      }
    return %r : tensor<32xi8>
  }
}
