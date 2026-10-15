module {
  func.func @main(%a: tensor<8x4xi8>, %b: tensor<8xi8>) -> tensor<8xi8> {
    %r = linalg.reduce ins(%a : tensor<8x4xi8>) outs(%b : tensor<8xi8>) dimensions = [1]
      (%in: i8, %init: i8) {
        %s = arith.maxsi %in, %init : i8
        linalg.yield %s : i8
      }
    return %r : tensor<8xi8>
  }
}
