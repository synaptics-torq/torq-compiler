module {
  func.func @main(%arg0: tensor<1x24x24x8xi8>) -> tensor<1x14x14x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 11, 11>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x24x24x8xi8>) -> tensor<1x14x14x8xi8>
    return %0 : tensor<1x14x14x8xi8>
  }
}
