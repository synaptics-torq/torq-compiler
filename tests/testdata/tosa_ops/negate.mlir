module {
  func.func @main(%arg0: tensor<8x16xi8>) -> tensor<8x16xi8> {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.negate %arg0, %cst, %cst_0 : (tensor<8x16xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<8x16xi8>
    return %0 : tensor<8x16xi8>
  }
}

