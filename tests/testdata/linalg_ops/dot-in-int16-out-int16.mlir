module {
  func.func @main(%arg0: tensor<3669xi16>, %arg1: tensor<3669xi16>) -> (tensor<i16>) {
    %init = tensor.empty() : tensor<i16>
    %cst = arith.constant 0 : i16
    %1 = linalg.fill ins(%cst : i16) outs(%init : tensor<i16>) -> tensor<i16>
    %2 = linalg.dot ins(%arg1, %arg0 : tensor<3669xi16>, tensor<3669xi16>) outs(%1 : tensor<i16>) -> tensor<i16>
    return %2 : tensor<i16>
  }
}
