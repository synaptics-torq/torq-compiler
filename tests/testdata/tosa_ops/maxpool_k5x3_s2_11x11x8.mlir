// 5x3 stride 2 with no padding (kernel borders different along H and W)
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x4x5x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 5, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>) -> tensor<1x4x5x8xi8>
    return %0 : tensor<1x4x5x8xi8>
  }
}
