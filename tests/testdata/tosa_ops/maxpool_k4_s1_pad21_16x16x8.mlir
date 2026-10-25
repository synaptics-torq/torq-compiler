// 4x4 stride 1 with padding 2 then 1: leading padding k / 2, on the NSS with the kernel border halves swapped
module {
  func.func @main(%arg0: tensor<1x16x16x8xi8>) -> tensor<1x16x16x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 4>, pad = array<i64: 2, 1, 2, 1>, stride = array<i64: 1, 1>} : (tensor<1x16x16x8xi8>) -> tensor<1x16x16x8xi8>
    return %0 : tensor<1x16x16x8xi8>
  }
}
