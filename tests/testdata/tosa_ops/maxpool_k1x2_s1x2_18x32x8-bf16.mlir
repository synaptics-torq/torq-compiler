// 1x2 stride [1, 2] on 18x32x8 bf16: swapped to the vertical kernel, whose rows have 18 elements
module {
  func.func @main(%arg0: tensor<1x18x32x8xbf16>) -> tensor<1x18x16x8xbf16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 2>} : (tensor<1x18x32x8xbf16>) -> tensor<1x18x16x8xbf16>
    return %0 : tensor<1x18x16x8xbf16>
  }
}
