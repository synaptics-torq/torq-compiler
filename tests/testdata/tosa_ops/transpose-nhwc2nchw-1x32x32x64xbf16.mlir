module {
  func.func @main(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x64x32x32xbf16> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x64xbf16>) -> tensor<1x64x32x32xbf16>
    return %0 : tensor<1x64x32x32xbf16>
  }
}
