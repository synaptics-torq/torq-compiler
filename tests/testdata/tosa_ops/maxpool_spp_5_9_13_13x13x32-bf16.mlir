module {
  func.func @main(%arg0: tensor<1x13x13x32xbf16>) -> (tensor<1x13x13x32xbf16>, tensor<1x13x13x32xbf16>, tensor<1x13x13x32xbf16>) {
    %0 = tosa.max_pool2d %arg0 kernel([5, 5]) stride([1, 1]) pad([2, 2, 2, 2]) : (tensor<1x13x13x32xbf16>) -> tensor<1x13x13x32xbf16>
    %1 = tosa.max_pool2d %arg0 kernel([9, 9]) stride([1, 1]) pad([4, 4, 4, 4]) : (tensor<1x13x13x32xbf16>) -> tensor<1x13x13x32xbf16>
    %2 = tosa.max_pool2d %arg0 kernel([13, 13]) stride([1, 1]) pad([6, 6, 6, 6]) : (tensor<1x13x13x32xbf16>) -> tensor<1x13x13x32xbf16>
    return %0, %1, %2 : tensor<1x13x13x32xbf16>, tensor<1x13x13x32xbf16>, tensor<1x13x13x32xbf16>
  }
}
