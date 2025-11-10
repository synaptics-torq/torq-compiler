module {
  func.func @main(%arg0: tensor<2x3x4x64x12x6xi16>) -> (tensor<2x3x4x64x12xi16>) {
    %cst = arith.constant -32768 : i16
    %0 = tensor.empty() : tensor<2x3x4x64x12xi16>
    %1 = linalg.fill ins(%cst : i16) outs(%0 : tensor<2x3x4x64x12xi16>) -> tensor<2x3x4x64x12xi16>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x3x4x64x12x6xi16>) outs(%1 : tensor<2x3x4x64x12xi16>) dimensions = [5]
      (%in: i16, %init: i16) {
        %11 = arith.maxsi %in, %init : i16
        linalg.yield %11 : i16
      }
    return %reduced : tensor<2x3x4x64x12xi16>
  }
}
