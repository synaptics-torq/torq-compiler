module {
  func.func @main(%arg0: tensor<1x12x25xi8>, %arg1: tensor<1x64x12xi8>) -> (tensor<1x64x25xi16>, tensor<1x64x25xi16>) {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.matmul %arg1, %arg0, %cst, %cst : (tensor<1x64x12xi8>, tensor<1x12x25xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x64x25xi16>
    return %0, %0 : tensor<1x64x25xi16>, tensor<1x64x25xi16>
  }
}

