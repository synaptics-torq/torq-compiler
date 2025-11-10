module {
  func.func @main(%arg0: tensor<2x5x2x16x8x2xi8>) -> (tensor<2x5x2x16x8x2xi8>) {
    %0 = tosa.identity %arg0 : (tensor<2x5x2x16x8x2xi8>) -> tensor<2x5x2x16x8x2xi8>
    return %0 : tensor<2x5x2x16x8x2xi8>
  }
}