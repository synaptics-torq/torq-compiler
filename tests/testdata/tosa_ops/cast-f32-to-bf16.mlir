module {
  func.func @main(%arg0: tensor<160xf32>) -> (tensor<160xbf16>) {
    %0 = tosa.cast %arg0 : (tensor<160xf32>) -> tensor<160xbf16>
    return %0 : tensor<160xbf16>
  }
}
