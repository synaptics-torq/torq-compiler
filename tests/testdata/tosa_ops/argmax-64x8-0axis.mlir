module {
  func.func @main(%arg0: tensor<64x8xi8>) -> tensor<8xi32> {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<64x8xi8>) -> tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}
