module {
  func.func @main(%arg0: tensor<2x64x4x8x3xbf16>) -> tensor<2x4x8x3x64xbf16> {
    %0 = tosa.transpose %arg0 perms([0, 2, 3, 4, 1]) : (tensor<2x64x4x8x3xbf16>) -> tensor<2x4x8x3x64xbf16>
    return %0 : tensor<2x4x8x3x64xbf16>
  }
}
