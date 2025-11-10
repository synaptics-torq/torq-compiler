module {
  func.func @main(%arg0: tensor<4x64xi1>, %arg1: tensor<4x64xi1>) -> tensor<4x64xi1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<4x64xi1>, tensor<4x64xi1>) -> tensor<4x64xi1>
    return %0 : tensor<4x64xi1>
  }
}
