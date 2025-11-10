module {
  func.func @main(%arg0: tensor<1x16x4x2xbf16>) -> (tensor<1x16x4x2xi8>) {
    %0 = tosa.cast %arg0 : (tensor<1x16x4x2xbf16>) -> tensor<1x16x4x2xi8>
    return %0 : tensor<1x16x4x2xi8>
  }
}
