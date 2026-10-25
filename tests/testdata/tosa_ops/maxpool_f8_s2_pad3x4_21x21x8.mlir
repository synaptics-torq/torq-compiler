module {
  func.func @main(%arg0: tensor<1x21x21x8xi8>) -> tensor<1x11x11x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 8, 8>, pad = array<i64: 3, 4, 3, 4>, stride = array<i64: 2, 2>} : (tensor<1x21x21x8xi8>) -> tensor<1x11x11x8xi8>
    return %0 : tensor<1x11x11x8xi8>
  }
}
