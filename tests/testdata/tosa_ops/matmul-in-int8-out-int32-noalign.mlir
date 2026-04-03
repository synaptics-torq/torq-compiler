module {
  func.func @main(%arg0: tensor<1x58x34xi8>, %arg1: tensor<1x34x68xi8>) -> tensor<1x58x68xi32> {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %cst, %cst_0 : (tensor<1x58x34xi8>, tensor<1x34x68xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x58x68xi32>
    return %0 : tensor<1x58x68xi32>
  }
}

