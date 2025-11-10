module {
  func.func @main(%arg0: tensor<1x4x16x16xi32>) -> (tensor<1x4x16x16xi32>) {
    %0 = tosa.identity %arg0 : (tensor<1x4x16x16xi32>) -> tensor<1x4x16x16xi32>
    return %0 : tensor<1x4x16x16xi32>
  }
}