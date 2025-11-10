module {
  func.func @main(%arg0: tensor<1x21x1xi32>, %arg1: tensor<1x21x1024xi32>) -> (tensor<1x21x1024xi32>) {
    %187 = tosa.sub %arg0, %arg1 : (tensor<1x21x1xi32>, tensor<1x21x1024xi32>) -> tensor<1x21x1024xi32>
    return %187 : tensor<1x21x1024xi32>
  }
}

