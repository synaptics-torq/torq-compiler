// 3x5 stride 2 with same padding (bf16)
module {
  func.func @main(%arg0: tensor<1x11x11x8xbf16>) -> tensor<1x6x6x8xbf16> {
    %0 = tosa.max_pool2d %arg0 kernel([3, 5]) stride([2, 2]) pad([1, 1, 2, 2]) : (tensor<1x11x11x8xbf16>) -> tensor<1x6x6x8xbf16>
    return %0 : tensor<1x6x6x8xbf16>
  }
}
