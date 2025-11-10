module {
  func.func @main(%arg0: tensor<64x128x6xi32>) -> (tensor<64x128xi32>) {
    %cst = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<64x128xi32>
    %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<64x128xi32>) -> tensor<64x128xi32>
    %reduced = linalg.reduce ins(%arg0 : tensor<64x128x6xi32>) outs(%0 : tensor<64x128xi32>) dimensions = [2] 
      (%in: i32, %init: i32) {
        %11 = arith.addi %in, %init : i32
        linalg.yield %11 : i32
      }
    return %reduced : tensor<64x128xi32>
  }
}
