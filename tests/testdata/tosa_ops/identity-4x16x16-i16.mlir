module {
  func.func @main(%arg0: tensor<1x4x16x16xi16>) -> (tensor<1x4x16x16xi16>) {
    %0 = tosa.identity %arg0 : (tensor<1x4x16x16xi16>) -> tensor<1x4x16x16xi16>
    return %0 : tensor<1x4x16x16xi16>
  }
}