module {
  func.func @main(%arg0: tensor<2x48x7x9xbf16>) -> tensor<2x7x9x48xbf16> {
    %0 = tosa.transpose %arg0 perms([0, 2, 3, 1]) : (tensor<2x48x7x9xbf16>) -> tensor<2x7x9x48xbf16>
    return %0 : tensor<2x7x9x48xbf16>
  }
}
