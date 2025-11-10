module {
  func.func @main(%arg0: tensor<2x3x4x2x12x6xi32>) -> (tensor<2x3x4x2x12xi32>) {
    %cst = arith.constant -1 : i32
    %0 = tensor.empty() : tensor<2x3x4x2x12xi32>
    %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x3x4x2x12xi32>) -> tensor<2x3x4x2x12xi32>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x3x4x2x12x6xi32>) outs(%1 : tensor<2x3x4x2x12xi32>) dimensions = [5]
      (%in: i32, %init: i32) {
        %11 = arith.andi %in, %init : i32
        linalg.yield %11 : i32
      }
    return %reduced : tensor<2x3x4x2x12xi32>
  }
}
