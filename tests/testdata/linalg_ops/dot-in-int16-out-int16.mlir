module {
  func.func @main(%arg0: tensor<3669xi16>, %arg1: tensor<3669xi16>) -> (tensor<i16>) {
    %init = tensor.empty() : tensor<i16>
    %0 = linalg.dot ins(%arg1, %arg0 : tensor<3669xi16>, tensor<3669xi16>) outs(%init : tensor<i16>) -> tensor<i16>
    return %0 : tensor<i16>
  }
}
