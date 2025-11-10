module {
  func.func @main(%arg0: tensor<1x4x64xi8>) -> (tensor<1x64xi32>) {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<1x4x64xi8>) -> tensor<1x64xi32>
    return %0 : tensor<1x64xi32>
  }
}