module {
  func.func @main(%arg0: tensor<1x7x128xi8>) -> (tensor<1x128xi32>) {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<1x7x128xi8>) -> tensor<1x128xi32>
    return %0 : tensor<1x128xi32>
  }
}