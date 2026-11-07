// 2x2 stride 1 with padding 1 then 0 (SAME_LOWER): leading padding k / 2, on the NSS with the kernel border halves swapped
module {
  func.func @main(%arg0: tensor<1x16x16x8xi8>) -> tensor<1x16x16x8xi8> {
    %0 = tosa.max_pool2d %arg0 kernel([2, 2]) stride([1, 1]) pad([1, 0, 1, 0]) : (tensor<1x16x16x8xi8>) -> tensor<1x16x16x8xi8>
    return %0 : tensor<1x16x16x8xi8>
  }
}
