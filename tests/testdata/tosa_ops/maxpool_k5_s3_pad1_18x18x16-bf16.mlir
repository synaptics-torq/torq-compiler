// 5x5 stride 3 with padding 1 (bf16): split in a 3x3 stride-1 maxpool and two 1D stride-3 maxpools
module {
  func.func @main(%arg0: tensor<1x18x18x16xbf16>) -> tensor<1x6x6x16xbf16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 5, 5>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 3, 3>} : (tensor<1x18x18x16xbf16>) -> tensor<1x6x6x16xbf16>
    return %0 : tensor<1x6x6x16xbf16>
  }
}
