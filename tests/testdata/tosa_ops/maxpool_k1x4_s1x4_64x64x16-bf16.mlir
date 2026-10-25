module {
  func.func @main(%arg0: tensor<1x64x64x16xbf16>) -> tensor<1x64x16x16xbf16> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 4>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 4>} : (tensor<1x64x64x16xbf16>) -> tensor<1x64x16x16xbf16>
    return %0 : tensor<1x64x16x16xbf16>
  }
}
