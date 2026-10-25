// Not supported on the NSS: runs on CSS
module {
  func.func @main(%arg0: tensor<1x17x17x8xi8>) -> tensor<1x5x5x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 5, 5>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 3, 3>} : (tensor<1x17x17x8xi8>) -> tensor<1x5x5x8xi8>
    return %0 : tensor<1x5x5x8xi8>
  }
}
