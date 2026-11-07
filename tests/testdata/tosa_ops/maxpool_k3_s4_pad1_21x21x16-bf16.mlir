module {
  func.func @main(%arg0: tensor<1x21x21x16xbf16>) -> tensor<1x6x6x16xbf16> {
    %0 = tosa.max_pool2d %arg0 kernel([3, 3]) stride([4, 4]) pad([1, 1, 1, 1]) : (tensor<1x21x21x16xbf16>) -> tensor<1x6x6x16xbf16>
    return %0 : tensor<1x6x6x16xbf16>
  }
}
