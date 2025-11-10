module {
  func.func @main(%arg0: tensor<16xi32>) -> (tensor<16xbf16>) {
    %0 = tosa.cast %arg0 : (tensor<16xi32>) -> tensor<16xbf16>
    return %0 : tensor<16xbf16>
  }
}
