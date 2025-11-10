module {
  func.func @main(%arg0: tensor<2x3x4x8x12x6xi32>) -> tensor<2x3x4x8x12xi32> {
    %c127_i32 = arith.constant 2147483647 : i32
    %0 = tensor.empty() : tensor<2x3x4x8x12xi32>
    %1 = linalg.fill ins(%c127_i32 : i32) outs(%0 : tensor<2x3x4x8x12xi32>) -> tensor<2x3x4x8x12xi32>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x3x4x8x12x6xi32>) outs(%1 : tensor<2x3x4x8x12xi32>) dimensions = [5] 
      (%in: i32, %init: i32) {
        %2 = arith.minsi %in, %init : i32
        linalg.yield %2 : i32
      }
    return %reduced : tensor<2x3x4x8x12xi32>
  }
}

