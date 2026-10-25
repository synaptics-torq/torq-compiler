// 6x6 stride 2 with no padding (leading padding 0, not (k - 1) / 2)
module {
  func.func @main(%arg0: tensor<1x20x20x8xi8>) -> tensor<1x8x8x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 6, 6>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x20x20x8xi8>) -> tensor<1x8x8x8xi8>
    return %0 : tensor<1x8x8x8xi8>
  }
}
