module {
  func.func @main(%arg0: tensor<128x256xi8>, %arg1: tensor<64x128xi8>) -> (tensor<64x256xi16>) {
    %init = tensor.empty() : tensor<64x256xi16>
    %0 = linalg.matmul ins(%arg1, %arg0 : tensor<64x128xi8>, tensor<128x256xi8>) outs(%init : tensor<64x256xi16>) -> tensor<64x256xi16>
    return %0 : tensor<64x256xi16>
  }
}

