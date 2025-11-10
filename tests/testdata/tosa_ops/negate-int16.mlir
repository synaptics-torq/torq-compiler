module {
  func.func @main(%arg0: tensor<8x13x9xi16>) -> (tensor<8x13x9xi16>) {
    %0 = tosa.negate %arg0 : (tensor<8x13x9xi16>) -> tensor<8x13x9xi16>
    return %0 : tensor<8x13x9xi16>
  }
}
