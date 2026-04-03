module {
  func.func @main(%arg0: tensor<128x64xi8>) -> tensor<64x128xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 0>} : (tensor<128x64xi8>) -> tensor<64x128xi8>
    return %0 : tensor<64x128xi8>
  }
}

