module {
  func.func @main(%arg0: tensor<1x160xf32>) -> (tensor<1x160xi8>) {
    %0 = tosa.cast %arg0 : (tensor<1x160xf32>) -> tensor<1x160xi8>
    return %0 : tensor<1x160xi8>
  }
}
