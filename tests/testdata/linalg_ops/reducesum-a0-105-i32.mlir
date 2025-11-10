module {
  func.func @main(%arg0: tensor<6x64x105xi32>) -> (tensor<64x105xi32>) {
    %cst = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<64x105xi32>
    %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<64x105xi32>) -> tensor<64x105xi32>
    %reduced = linalg.reduce ins(%arg0 : tensor<6x64x105xi32>) outs(%1 : tensor<64x105xi32>) dimensions = [0] 
      (%in: i32, %init: i32) {
        %11 = arith.addi %in, %init : i32
        linalg.yield %11 : i32
      }
    return %reduced : tensor<64x105xi32>
  }
}
