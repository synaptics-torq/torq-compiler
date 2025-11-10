module {
  func.func @main(%arg0: tensor<1x64x8xi8>) -> (tensor<1x64xi32>) {
    %0 = tosa.argmax %arg0 {axis = 2 : i32} : (tensor<1x64x8xi8>) -> tensor<1x64xi32>
    return %0 : tensor<1x64xi32>
  }
}