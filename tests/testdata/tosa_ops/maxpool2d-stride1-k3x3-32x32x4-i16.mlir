module {
  func.func @main(%arg0: tensor<1x32x32x4xi16>) -> (tensor<1x32x32x4xi16>) {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x4xi16>) -> tensor<1x32x32x4xi16>
    return %0 : tensor<1x32x32x4xi16>
  }
}

