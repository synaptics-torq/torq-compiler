module {
  func.func @main(%arg0: tensor<4x64xi1>) -> tensor<4x64xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<4x64xi1>) -> tensor<4x64xi1>
    return %0 : tensor<4x64xi1>
  }
}
