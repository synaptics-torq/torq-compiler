module {
  func.func @main(%arg0: tensor<2x64x4x8x3xi8>) -> tensor<2x4x8x3x64xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 4, 1>} : (tensor<2x64x4x8x3xi8>) -> tensor<2x4x8x3x64xi8>
    return %0 : tensor<2x4x8x3x64xi8>
  }
}
