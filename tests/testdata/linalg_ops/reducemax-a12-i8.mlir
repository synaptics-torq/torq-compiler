module {
  func.func @main(%a: tensor<16x32x128xi8>) -> tensor<16xi8> {
    %c = arith.constant -128 : i8
    %e = tensor.empty() : tensor<16xi8>
    %f = linalg.fill ins(%c : i8) outs(%e : tensor<16xi8>) -> tensor<16xi8>
    %r = linalg.reduce ins(%a : tensor<16x32x128xi8>) outs(%f : tensor<16xi8>) dimensions = [1, 2]
      (%in: i8, %init: i8) {
        %s = arith.maxsi %in, %init : i8
        linalg.yield %s : i8
      }
    return %r : tensor<16xi8>
  }
}
