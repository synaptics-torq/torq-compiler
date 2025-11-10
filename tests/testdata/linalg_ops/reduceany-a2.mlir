module {
  func.func @main(%arg0: tensor<2x64x5xi16>) -> (tensor<2x64xi16>) {
    %0 = tensor.empty() : tensor<2x64xi16>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x64x5xi16>) outs(%0 : tensor<2x64xi16>) dimensions = [2] 
      (%in: i16, %init: i16) {
        %11 = arith.ori %in, %init : i16
        linalg.yield %11 : i16
      }
    return %reduced : tensor<2x64xi16>
  }
}
