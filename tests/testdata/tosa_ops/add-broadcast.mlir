module {
  func.func @main(%arg0: tensor<1x21x1024xi32>, %arg1: tensor<1x1x1xi32>) -> (tensor<1x21x1024xi32>) {
    %189 = tosa.add %arg0, %arg1 : (tensor<1x21x1024xi32>, tensor<1x1x1xi32>) -> tensor<1x21x1024xi32>
    return %189 : tensor<1x21x1024xi32>
  }
}

