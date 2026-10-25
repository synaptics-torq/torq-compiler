module {
  func.func @main(%arg0: tensor<33x40x64xi8>) -> tensor<64x33x40xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<33x40x64xi8>) -> tensor<64x33x40xi8>
    return %0 : tensor<64x33x40xi8>
  }
}
