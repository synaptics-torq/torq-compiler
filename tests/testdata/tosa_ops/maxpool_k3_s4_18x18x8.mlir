module {
  func.func @main(%arg0: tensor<1x18x18x8xi8>) -> tensor<1x5x5x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 4, 4>} : (tensor<1x18x18x8xi8>) -> tensor<1x5x5x8xi8>
    return %0 : tensor<1x5x5x8xi8>
  }
}
