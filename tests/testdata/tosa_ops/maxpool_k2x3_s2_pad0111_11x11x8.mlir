// 2x3 stride 2 with padding 0 then 1 along H, 1 then 1 along W
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 3>, pad = array<i64: 0, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8>
    return %0 : tensor<1x6x6x8xi8>
  }
}
