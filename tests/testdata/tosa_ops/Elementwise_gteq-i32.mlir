module {
  func.func @main(%arg0: tensor<4x64xi32>, %arg1: tensor<4x64xi32>) -> tensor<4x64xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<4x64xi32>, tensor<4x64xi32>) -> tensor<4x64xi1>
    return %0 : tensor<4x64xi1>
  }
}
