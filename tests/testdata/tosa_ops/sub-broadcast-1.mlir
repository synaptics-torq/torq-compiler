module {
  func.func @main(%arg0: tensor<1x21x1xi32>, %arg1: tensor<1x1x1xi32>) -> (tensor<1x21x1xi32>) {
    %195 = tosa.sub %arg0, %arg1 : (tensor<1x21x1xi32>, tensor<1x1x1xi32>) -> tensor<1x21x1xi32>
    return %195 : tensor<1x21x1xi32>
  }
}

