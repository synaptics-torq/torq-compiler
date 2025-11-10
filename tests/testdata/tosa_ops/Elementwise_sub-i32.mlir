module {
  func.func @main(%arg0: tensor<4x64x64xi32>, %arg1: tensor<4x64x64xi32>) -> tensor<4x64x64xi32> {
    %197 = tosa.sub %arg0, %arg1 : (tensor<4x64x64xi32>, tensor<4x64x64xi32>) -> tensor<4x64x64xi32>
    return %197 : tensor<4x64x64xi32>
  }
}
