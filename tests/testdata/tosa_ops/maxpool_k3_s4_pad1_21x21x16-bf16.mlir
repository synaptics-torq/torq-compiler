module {
  func.func @main(%arg0: tensor<1x21x21x16xbf16>) -> tensor<1x6x6x16xbf16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 4, 4>} : (tensor<1x21x21x16xbf16>) -> tensor<1x6x6x16xbf16>
    return %0 : tensor<1x6x6x16xbf16>
  }
}
