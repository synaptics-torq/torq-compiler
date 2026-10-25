// Stride 4 with a 4x4 kernel and no padding: split in a 4x4 stride-2 and a 1x1 stride-2 maxpool
module {
  func.func @main(%arg0: tensor<1x20x20x8xi8>) -> tensor<1x5x5x8xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 4>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 4, 4>} : (tensor<1x20x20x8xi8>) -> tensor<1x5x5x8xi8>
    return %0 : tensor<1x5x5x8xi8>
  }
}
