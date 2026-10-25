// 9x9 stride 3: split in a 7x7 stride-1 maxpool and two 1D stride-3 maxpools
module {
  func.func @main(%arg0: tensor<1x21x21x8xi8>) -> tensor<1x5x5x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 9, 9>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 3, 3>} : (tensor<1x21x21x8xi8>) -> tensor<1x5x5x8xi8>
    return %0 : tensor<1x5x5x8xi8>
  }
}
