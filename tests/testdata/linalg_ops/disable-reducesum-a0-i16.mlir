module {
  func.func @main(%arg0: tensor<6x64x128xi16>) -> (tensor<64x128xi16>) {
    %cst = arith.constant 0 : i16
    %0 = tensor.empty() : tensor<64x128xi16>
    %1 = linalg.fill ins(%cst : i16) outs(%0 : tensor<64x128xi16>) -> tensor<64x128xi16>
    %reduced = linalg.reduce ins(%arg0 : tensor<6x64x128xi16>) outs(%1 : tensor<64x128xi16>) dimensions = [0] 
      (%in: i16, %init: i16) {
        %11 = arith.addi %in, %init : i16
        linalg.yield %11 : i16
      }
    return %reduced : tensor<64x128xi16>
  }
}
