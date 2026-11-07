// 2x1 stride [2, 1] on 16x24x8 bf16: vertical 1D kernel with rows of 24 elements, more than one
// ACT vector of 16
module {
  func.func @main(%arg0: tensor<1x16x24x8xbf16>) -> tensor<1x8x24x8xbf16> {
    %0 = tosa.max_pool2d %arg0 kernel([2, 1]) stride([2, 1]) pad([0, 0, 0, 0]) : (tensor<1x16x24x8xbf16>) -> tensor<1x8x24x8xbf16>
    return %0 : tensor<1x8x24x8xbf16>
  }
}
