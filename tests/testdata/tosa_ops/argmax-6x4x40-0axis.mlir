module {
  func.func @main(%arg0: tensor<6x4x40xi8>) -> tensor<4x40xi32> {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<6x4x40xi8>) -> tensor<4x40xi32>
    return %0 : tensor<4x40xi32>
  }
}
