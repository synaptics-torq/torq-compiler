module {
  func.func @main(%arg0: tensor<8x128x256xi8>, %arg1: tensor<8x64x128xi8>) -> (tensor<8x64x256xi16>) {
    %init = tensor.empty() : tensor<8x64x256xi16>
    %0 = linalg.batch_matmul ins(%arg1, %arg0 : tensor<8x64x128xi8>, tensor<8x128x256xi8>) outs(%init : tensor<8x64x256xi16>) -> tensor<8x64x256xi16>
    return %0 : tensor<8x64x256xi16>
  }
}

