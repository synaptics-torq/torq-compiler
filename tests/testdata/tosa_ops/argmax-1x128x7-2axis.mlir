module {
  func.func @main(%arg0: tensor<1x128x7xi8>) -> (tensor<1x128xi32>) {
    %0 = tosa.argmax %arg0 {axis = 2 : i32} : (tensor<1x128x7xi8>) -> tensor<1x128xi32>
    return %0 : tensor<1x128xi32>
  }
}