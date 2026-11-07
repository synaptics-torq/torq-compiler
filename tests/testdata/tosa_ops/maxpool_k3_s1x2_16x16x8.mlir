// 3x3 with strides [1, 2]: split in a 3x2 stride-1 maxpool and a 1x2 stride-[1, 2] maxpool
module {
  func.func @main(%arg0: tensor<1x16x16x8xi8>) -> tensor<1x16x8x8xi8> {
    %0 = tosa.max_pool2d %arg0 kernel([3, 3]) stride([1, 2]) pad([1, 1, 1, 0]) : (tensor<1x16x16x8xi8>) -> tensor<1x16x8x8xi8>
    return %0 : tensor<1x16x8x8xi8>
  }
}
