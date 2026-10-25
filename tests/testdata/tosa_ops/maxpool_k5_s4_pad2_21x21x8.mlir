module {
  func.func @main(%arg0: tensor<1x21x21x8xi8>) -> tensor<1x6x6x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 5, 5>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 4, 4>} : (tensor<1x21x21x8xi8>) -> tensor<1x6x6x8xi8>
    return %0 : tensor<1x6x6x8xi8>
  }
}
