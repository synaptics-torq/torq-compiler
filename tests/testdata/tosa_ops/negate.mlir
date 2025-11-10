module {
  func.func @main(%arg0: tensor<8x16xi8>) -> (tensor<8x16xi8>) {
    %0 = tosa.negate %arg0 : (tensor<8x16xi8>) -> tensor<8x16xi8>
    return %0 : tensor<8x16xi8>
  }
}