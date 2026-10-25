module {
  func.func @main(%arg0: tensor<1x16x256x64xi8>) -> tensor<1x16x32x64xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 8>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 8>} : (tensor<1x16x256x64xi8>) -> tensor<1x16x32x64xi8>
    return %0 : tensor<1x16x32x64xi8>
  }
}
