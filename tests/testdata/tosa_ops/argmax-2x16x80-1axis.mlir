module {
  func.func @main(%arg0: tensor<2x16x80xi8>) -> tensor<2x80xi32> {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<2x16x80xi8>) -> tensor<2x80xi32>
    return %0 : tensor<2x80xi32>
  }
}
