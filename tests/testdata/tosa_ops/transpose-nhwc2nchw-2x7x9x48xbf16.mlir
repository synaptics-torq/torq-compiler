module {
  func.func @main(%arg0: tensor<2x7x9x48xbf16>) -> tensor<2x48x7x9xbf16> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x7x9x48xbf16>) -> tensor<2x48x7x9xbf16>
    return %0 : tensor<2x48x7x9xbf16>
  }
}
