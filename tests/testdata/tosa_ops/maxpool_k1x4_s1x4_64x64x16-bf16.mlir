module {
  func.func @main(%arg0: tensor<1x64x64x16xbf16>) -> tensor<1x64x16x16xbf16> {
    %0 = tosa.max_pool2d %arg0 kernel([1, 4]) stride([1, 4]) pad([0, 0, 0, 0]) : (tensor<1x64x64x16xbf16>) -> tensor<1x64x16x16xbf16>
    return %0 : tensor<1x64x16x16xbf16>
  }
}
