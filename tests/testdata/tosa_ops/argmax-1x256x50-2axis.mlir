module {
  func.func @main(%arg0: tensor<1x256x50xi8>) -> (tensor<1x256xi32>) {
    %0 = tosa.argmax %arg0 {axis = 2 : i32} : (tensor<1x256x50xi8>) -> tensor<1x256xi32>
    return %0 : tensor<1x256xi32>
  }
}