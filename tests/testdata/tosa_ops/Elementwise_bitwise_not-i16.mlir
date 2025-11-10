module {
  func.func @main(%arg0: tensor<4x64xi16>) -> tensor<4x64xi16> {
    %0 = tosa.bitwise_not %arg0 : (tensor<4x64xi16>) -> tensor<4x64xi16>
    return %0 : tensor<4x64xi16>
  }
}
