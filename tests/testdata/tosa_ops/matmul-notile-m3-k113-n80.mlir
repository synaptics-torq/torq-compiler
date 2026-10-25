module {
  func.func @main(%arg0: tensor<2x113x80xi8>, %arg1: tensor<2x3x113xi8>) -> tensor<2x3x80xi16> {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.matmul %arg1, %arg0, %cst, %cst_0 : (tensor<2x3x113xi8>, tensor<2x113x80xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x3x80xi16>
    return %0 : tensor<2x3x80xi16>
  }
}

