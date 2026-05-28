module {
  func.func @main(%arg0: tensor<8x128x256xi8>, %arg1: tensor<8x64x128xi8>) -> (tensor<8x64x256xi16>) {
    %cst = arith.constant 0 : i16
    %0 = tensor.empty() : tensor<8x64x256xi16>
    %1 = linalg.fill ins(%cst : i16) outs(%0 : tensor<8x64x256xi16>) -> tensor<8x64x256xi16>
    %2 = linalg.batch_matmul ins(%arg1, %arg0 : tensor<8x64x128xi8>, tensor<8x128x256xi8>) outs(%1 : tensor<8x64x256xi16>) -> tensor<8x64x256xi16>
    return %2 : tensor<8x64x256xi16>
  }
}

