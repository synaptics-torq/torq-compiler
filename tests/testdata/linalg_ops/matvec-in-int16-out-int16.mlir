module {
  func.func @main(%arg0: tensor<128x64xi16>, %arg1: tensor<64xi16>) -> (tensor<128xi16>) {
    %init = tensor.empty() : tensor<128xi16>
    %0 = linalg.matvec ins(%arg0, %arg1 : tensor<128x64xi16>, tensor<64xi16>) outs(%init : tensor<128xi16>) -> tensor<128xi16>
    return %0 : tensor<128xi16>
  }
}
