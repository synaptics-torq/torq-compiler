module {
  func.func @main(%arg0: tensor<1x20x20x8xbf16>) -> tensor<1x20x20x8xbf16> {
    %0 = tosa.max_pool2d %arg0 kernel([13, 13]) stride([1, 1]) pad([6, 6, 6, 6]) : (tensor<1x20x20x8xbf16>) -> tensor<1x20x20x8xbf16>
    return %0 : tensor<1x20x20x8xbf16>
  }
}
