module {
  func.func @main(%a: tensor<4x512x256xi8>) -> tensor<4xi8> {
    %c = arith.constant 0 : i8
    %e = tensor.empty() : tensor<4xi8>
    %f = linalg.fill ins(%c : i8) outs(%e : tensor<4xi8>) -> tensor<4xi8>
    %r = linalg.reduce ins(%a : tensor<4x512x256xi8>) outs(%f : tensor<4xi8>) dimensions = [1, 2]
      (%in: i8, %init: i8) {
        %s = arith.addi %in, %init : i8
        linalg.yield %s : i8
      }
    return %r : tensor<4xi8>
  }
}
