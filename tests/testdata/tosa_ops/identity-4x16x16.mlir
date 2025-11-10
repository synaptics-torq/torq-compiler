module {
  func.func @main(%arg0: tensor<1x4x16x16xi8>) -> (tensor<1x4x16x16xi8>) {
    %0 = tosa.identity %arg0 : (tensor<1x4x16x16xi8>) -> tensor<1x4x16x16xi8>
    return %0 : tensor<1x4x16x16xi8>
  }
}