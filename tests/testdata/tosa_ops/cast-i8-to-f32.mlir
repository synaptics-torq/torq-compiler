module {
  func.func @main(%arg0: tensor<16000xi8>) -> (tensor<16000xf32>) {
    %0 = tosa.cast %arg0 : (tensor<16000xi8>) -> tensor<16000xf32>
    return %0 : tensor<16000xf32>
  }
}
