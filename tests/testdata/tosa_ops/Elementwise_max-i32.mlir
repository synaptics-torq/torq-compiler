module {
  func.func @main(%arg0: tensor<4x64xi32>, %arg1: tensor<4x64xi32>) -> tensor<4x64xi32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<4x64xi32>, tensor<4x64xi32>) -> tensor<4x64xi32>
    return %0 : tensor<4x64xi32>
  }
}