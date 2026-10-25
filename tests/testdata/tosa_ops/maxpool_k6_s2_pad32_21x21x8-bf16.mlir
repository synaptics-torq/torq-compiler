// 6x6 stride 2 with padding 3 then 2 (bf16, leading padding k / 2)
module {
  func.func @main(%arg0: tensor<1x21x21x8xbf16>) -> tensor<1x11x11x8xbf16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 6, 6>, pad = array<i64: 3, 2, 3, 2>, stride = array<i64: 2, 2>} : (tensor<1x21x21x8xbf16>) -> tensor<1x11x11x8xbf16>
    return %0 : tensor<1x11x11x8xbf16>
  }
}
