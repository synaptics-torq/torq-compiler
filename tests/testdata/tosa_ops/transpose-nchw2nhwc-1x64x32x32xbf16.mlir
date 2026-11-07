module {
  func.func @main(%arg0: tensor<1x64x32x32xbf16>) -> tensor<1x32x32x64xbf16> {
    %0 = tosa.transpose %arg0 perms([0, 2, 3, 1]) : (tensor<1x64x32x32xbf16>) -> tensor<1x32x32x64xbf16>
    return %0 : tensor<1x32x32x64xbf16>
  }
}
