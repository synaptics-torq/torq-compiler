module {
  func.func @main(%arg0: tensor<128x64xi8>, %arg1: tensor<64xi8>) -> (tensor<128xi16>) {
    %init = tensor.empty() : tensor<128xi16>
    %0 = linalg.matvec ins(%arg0, %arg1 : tensor<128x64xi8>, tensor<64xi8>) outs(%init : tensor<128xi16>) -> tensor<128xi16>
    return %0 : tensor<128xi16>
  }
}
