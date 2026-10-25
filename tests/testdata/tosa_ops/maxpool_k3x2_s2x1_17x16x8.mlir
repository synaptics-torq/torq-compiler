// 3x2 with strides [2, 1] and no padding: split in a 2x2 stride-1 maxpool and a 2x1 stride-[2, 1] maxpool
module {
  func.func @main(%arg0: tensor<1x17x16x8xi8>) -> tensor<1x8x15x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 1>} : (tensor<1x17x16x8xi8>) -> tensor<1x8x15x8xi8>
    return %0 : tensor<1x8x15x8xi8>
  }
}
