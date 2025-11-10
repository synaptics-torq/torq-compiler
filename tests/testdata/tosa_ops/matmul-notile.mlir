module {
  func.func @main(%arg0: tensor<1x12x25xi8>, %arg1: tensor<1x64x12xi8>) -> (tensor<1x64x25xi16>) {
    %0 = tosa.matmul %arg1, %arg0 : (tensor<1x64x12xi8>, tensor<1x12x25xi8>) -> tensor<1x64x25xi16>
    return %0 : tensor<1x64x25xi16>
  }
}

