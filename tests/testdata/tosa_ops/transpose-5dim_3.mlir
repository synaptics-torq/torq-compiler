module {
  func.func @main(%arg0: tensor<1x2x64x2x36xi8>) -> tensor<1x2x2x36x64xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 1, 3, 4, 2>} : (tensor<1x2x64x2x36xi8>) -> tensor<1x2x2x36x64xi8>
    return %0 : tensor<1x2x2x36x64xi8>
  }
}

