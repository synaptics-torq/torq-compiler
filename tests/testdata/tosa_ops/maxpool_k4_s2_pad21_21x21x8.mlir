// 4x4 stride 2 with padding 2 then 1 (leading padding k / 2, not (k - 1) / 2)
module {
  func.func @main(%arg0: tensor<1x21x21x8xi8>) -> tensor<1x11x11x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 4>, pad = array<i64: 2, 1, 2, 1>, stride = array<i64: 2, 2>} : (tensor<1x21x21x8xi8>) -> tensor<1x11x11x8xi8>
    return %0 : tensor<1x11x11x8xi8>
  }
}
