module {
  func.func @main(%arg0: tensor<128xi8>, %arg1: tensor<128xi8>) -> (tensor<i16>) {
    %init = tensor.empty() : tensor<i16>
    %0 = linalg.dot ins(%arg1, %arg0 : tensor<128xi8>, tensor<128xi8>) outs(%init : tensor<i16>) -> tensor<i16>
    return %0 : tensor<i16>
  }
}
