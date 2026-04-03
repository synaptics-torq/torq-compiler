module {
  func.func @main(%arg0: tensor<1x68xi8>, %arg1: tensor<1x68xi8>) -> tensor<1x68xi32> {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.mul %arg0, %arg1, %cst : (tensor<1x68xi8>, tensor<1x68xi8>, tensor<1xi8>) -> tensor<1x68xi32>
    return %0 : tensor<1x68xi32>
  }
}

