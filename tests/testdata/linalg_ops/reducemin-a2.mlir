module {
  func.func @main(%arg0: tensor<64x128x6xi8>) -> tensor<64x128xi8> {
    %c127_i8 = arith.constant 127 : i8
    %0 = tensor.empty() : tensor<64x128xi8>
    %1 = linalg.fill ins(%c127_i8 : i8) outs(%0 : tensor<64x128xi8>) -> tensor<64x128xi8>
    %reduced = linalg.reduce ins(%arg0 : tensor<64x128x6xi8>) outs(%1 : tensor<64x128xi8>) dimensions = [2]
      (%in: i8, %init: i8) {
        %2 = arith.minsi %in, %init : i8
        linalg.yield %2 : i8
      }
    return %reduced : tensor<64x128xi8>
  }
}

