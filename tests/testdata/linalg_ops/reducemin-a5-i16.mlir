module {
  func.func @main(%arg0: tensor<2x3x4x64x12x6xi16>) -> tensor<2x3x4x64x12xi16> {
    %c127_i16 = arith.constant 32767 : i16
    %0 = tensor.empty() : tensor<2x3x4x64x12xi16>
    %1 = linalg.fill ins(%c127_i16 : i16) outs(%0 : tensor<2x3x4x64x12xi16>) -> tensor<2x3x4x64x12xi16>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x3x4x64x12x6xi16>) outs(%1 : tensor<2x3x4x64x12xi16>) dimensions = [5] 
      (%in: i16, %init: i16) {
        %2 = arith.minsi %in, %init : i16
        linalg.yield %2 : i16
      }
    return %reduced : tensor<2x3x4x64x12xi16>
  }
}

