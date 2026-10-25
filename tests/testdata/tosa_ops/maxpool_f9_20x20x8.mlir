module {
  func.func @main(%arg0: tensor<1x20x20x8xi8>) -> tensor<1x20x20x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 9, 9>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 1, 1>} : (tensor<1x20x20x8xi8>) -> tensor<1x20x20x8xi8>
    return %0 : tensor<1x20x20x8xi8>
  }
}
