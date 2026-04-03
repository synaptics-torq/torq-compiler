module {
  func.func @maxpool_test_batch1_2x2_stride2x2_32x16x4(%arg0: tensor<1x16x4x32xi8>) -> tensor<1x8x2x32xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x16x4x32xi8>) -> tensor<1x8x2x32xi8>
    return %0 : tensor<1x8x2x32xi8>
  }
}

