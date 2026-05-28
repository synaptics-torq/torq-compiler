module {
  func.func @main(%arg0: tensor<128x64xi16>, %arg1: tensor<64xi16>) -> (tensor<128xi16>) {
    %cst = arith.constant 0 : i16
    %0 = tensor.empty() : tensor<128xi16>
    %1 = linalg.fill ins(%cst : i16) outs(%0 : tensor<128xi16>) -> tensor<128xi16>
    %2 = linalg.matvec ins(%arg0, %arg1 : tensor<128x64xi16>, tensor<64xi16>) outs(%1 : tensor<128xi16>) -> tensor<128xi16>
    return %2 : tensor<128xi16>
  }
}
