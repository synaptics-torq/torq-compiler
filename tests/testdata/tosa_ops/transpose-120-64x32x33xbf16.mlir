module {
  func.func @main(%arg0: tensor<64x32x33xbf16>) -> tensor<32x33x64xbf16> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 2, 0>} : (tensor<64x32x33xbf16>) -> tensor<32x33x64xbf16>
    return %0 : tensor<32x33x64xbf16>
  }
}
