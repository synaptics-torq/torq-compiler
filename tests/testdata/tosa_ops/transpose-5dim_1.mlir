module {
  func.func @main(%arg0: tensor<1x1x64x32x32xi8>) -> tensor<1x1x32x64x32xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 1, 4, 2, 3>} : (tensor<1x1x64x32x32xi8>) -> tensor<1x1x32x64x32xi8>
    return %0 : tensor<1x1x32x64x32xi8>
  }
}

