module {
  func.func @main(%arg0: tensor<1x32x32x4xi8>) -> tensor<1x32x32x4xi8> {
    %0 = tosa.max_pool2d %arg0 kernel([3, 3]) stride([1, 1]) pad([1, 1, 1, 1]) : (tensor<1x32x32x4xi8>) -> tensor<1x32x32x4xi8>
    return %0 : tensor<1x32x32x4xi8>
  }
}

