module {
  func.func @main(%arg0: tensor<1x21x21x8xi8>) -> (tensor<1x21x21x8xi8>, tensor<1x11x11x8xi8>) {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x21x21x8xi8>) -> tensor<1x21x21x8xi8>
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 5, 5>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>} : (tensor<1x21x21x8xi8>) -> tensor<1x11x11x8xi8>
    return %0, %1 : tensor<1x21x21x8xi8>, tensor<1x11x11x8xi8>
  }
}
