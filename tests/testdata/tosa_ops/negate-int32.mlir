module {
  func.func @main(%arg0: tensor<1x8x16x3xi32>) -> (tensor<1x8x16x3xi32>) {
    %0 = tosa.negate %arg0 : (tensor<1x8x16x3xi32>) -> tensor<1x8x16x3xi32>
    return %0 : tensor<1x8x16x3xi32>
  }
}
