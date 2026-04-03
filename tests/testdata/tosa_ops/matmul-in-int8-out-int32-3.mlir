module {
  func.func @main(%arg0: tensor<1x1x3xi8>, %arg1: tensor<1x3x2xi8>) -> tensor<1x1x2xi32> {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %cst, %cst : (tensor<1x1x3xi8>, tensor<1x3x2xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x2xi32>
    return %0 : tensor<1x1x2xi32>
  }
}

