module {
  func.func @main(%arg0: tensor<1600xf32>) -> (tensor<1600xi16>) {
    %0 = tosa.cast %arg0 : (tensor<1600xf32>) -> tensor<1600xi16>
    return %0 : tensor<1600xi16>
  }
}
