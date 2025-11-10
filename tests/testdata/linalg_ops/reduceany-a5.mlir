module {
  func.func @main(%arg0: tensor<2x3x4x1x8x6xi32>) -> (tensor<2x3x4x1x8xi32>) {
    %0 = tensor.empty() : tensor<2x3x4x1x8xi32>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x3x4x1x8x6xi32>) outs(%0 : tensor<2x3x4x1x8xi32>) dimensions = [5]
      (%in: i32, %init: i32) {
        %11 = arith.ori %in, %init : i32
        linalg.yield %11 : i32
      }
    return %reduced : tensor<2x3x4x1x8xi32>
  }
}
