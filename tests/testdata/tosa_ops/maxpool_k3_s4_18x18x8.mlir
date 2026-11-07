module {
  func.func @main(%arg0: tensor<1x18x18x8xi8>) -> tensor<1x5x5x8xi8> {
    %0 = tosa.max_pool2d %arg0 kernel([3, 3]) stride([4, 4]) pad([1, 0, 1, 0]) : (tensor<1x18x18x8xi8>) -> tensor<1x5x5x8xi8>
    return %0 : tensor<1x5x5x8xi8>
  }
}
