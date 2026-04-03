module {
  func.func @main(%arg0: tensor<1x256x16x32xi8>) -> tensor<1x64x16x32xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 4, 1>} : (tensor<1x256x16x32xi8>) -> tensor<1x64x16x32xi8>
    return %0 : tensor<1x64x16x32xi8>
  }
}

