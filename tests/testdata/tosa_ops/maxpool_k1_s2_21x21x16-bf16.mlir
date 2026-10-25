module {
  func.func @main(%arg0: tensor<1x21x21x16xbf16>) -> tensor<1x11x11x16xbf16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x21x21x16xbf16>) -> tensor<1x11x11x16xbf16>
    return %0 : tensor<1x11x11x16xbf16>
  }
}
