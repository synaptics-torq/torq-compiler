// 3x3 stride 3 (non-overlapping): split in a 3x1 stride-[3, 1] and a 1x3 stride-[1, 3] maxpool
module {
  func.func @main(%arg0: tensor<1x18x18x8xi8>) -> tensor<1x6x6x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 3, 3>} : (tensor<1x18x18x8xi8>) -> tensor<1x6x6x8xi8>
    return %0 : tensor<1x6x6x8xi8>
  }
}
