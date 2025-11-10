module {
  func.func @main(%arg0: tensor<2x64xi8>) -> (tensor<64xi8>) {
    %cst = arith.constant 127 : i8
    %0 = tensor.empty() : tensor<64xi8>
    %1 = linalg.fill ins(%cst : i8) outs(%0 : tensor<64xi8>) -> tensor<64xi8>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x64xi8>) outs(%1 : tensor<64xi8>) dimensions = [0] 
      (%in: i8, %init: i8) {
        %11 = arith.andi %in, %init : i8
        linalg.yield %11 : i8
      }
    return %reduced : tensor<64xi8>
  }
}
