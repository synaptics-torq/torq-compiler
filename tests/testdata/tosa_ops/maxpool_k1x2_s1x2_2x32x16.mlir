module {
  func.func @main(%arg0: tensor<1x2x32x16xi8>) -> tensor<1x2x16x16xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 2>} : (tensor<1x2x32x16xi8>) -> tensor<1x2x16x16xi8>
    return %0 : tensor<1x2x16x16xi8>
  }
}
