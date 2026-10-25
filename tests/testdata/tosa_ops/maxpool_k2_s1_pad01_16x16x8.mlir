// 2x2 stride 1 with padding 0 then 1 (SAME_UPPER), on the NSS
module {
  func.func @main(%arg0: tensor<1x16x16x8xi8>) -> tensor<1x16x16x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x16x16x8xi8>) -> tensor<1x16x16x8xi8>
    return %0 : tensor<1x16x16x8xi8>
  }
}
