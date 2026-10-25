// 3x5 stride 2 with same padding (bf16)
module {
  func.func @main(%arg0: tensor<1x11x11x8xbf16>) -> tensor<1x6x6x8xbf16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 5>, pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xbf16>) -> tensor<1x6x6x8xbf16>
    return %0 : tensor<1x6x6x8xbf16>
  }
}
