// 3x3 with strides [2, 1]: split in a 2x3 stride-1 maxpool and a 2x1 stride-[2, 1] maxpool
module {
  func.func @main(%arg0: tensor<1x16x16x8xi8>) -> tensor<1x8x16x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 1>, stride = array<i64: 2, 1>} : (tensor<1x16x16x8xi8>) -> tensor<1x8x16x8xi8>
    return %0 : tensor<1x8x16x8xi8>
  }
}
