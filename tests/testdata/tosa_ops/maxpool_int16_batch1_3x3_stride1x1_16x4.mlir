module {
  func.func @maxpool_test_batch1_3x3_stride1x1_16x4(%arg0: tensor<1x16x4x32xi16>) -> tensor<1x16x4x32xi16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x16x4x32xi16>) -> tensor<1x16x4x32xi16>
    return %0 : tensor<1x16x4x32xi16>
  }
}

