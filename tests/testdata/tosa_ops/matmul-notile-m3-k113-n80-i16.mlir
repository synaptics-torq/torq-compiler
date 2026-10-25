module {
  func.func @main(%arg0: tensor<2x113x80xi16>, %arg1: tensor<2x3x113xi16>) -> tensor<2x3x80xi16> {
    %cst = arith.constant dense<0> : tensor<1xi16>
    %cst_0 = arith.constant dense<0> : tensor<1xi16>
    %0 = tosa.matmul %arg1, %arg0, %cst, %cst_0 : (tensor<2x3x113xi16>, tensor<2x113x80xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<2x3x80xi16>
    return %0 : tensor<2x3x80xi16>
  }
}

