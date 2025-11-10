module {
  func.func @main(%arg0: tensor<1x32x32x4xi8>) -> (tensor<1x16x16x4xi8>) {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x32x32x4xi8>) -> tensor<1x16x16x4xi8>
    return %0 : tensor<1x16x16x4xi8>
  }
}

