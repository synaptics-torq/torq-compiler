module {
  func.func @main(%arg0: tensor<1x20x20x8xbf16>) -> tensor<1x20x20x8xbf16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 13, 13>, pad = array<i64: 6, 6, 6, 6>, stride = array<i64: 1, 1>} : (tensor<1x20x20x8xbf16>) -> tensor<1x20x20x8xbf16>
    return %0 : tensor<1x20x20x8xbf16>
  }
}
