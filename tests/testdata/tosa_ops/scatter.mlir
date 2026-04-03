module {
  func.func @main(%arg0: tensor<1x256x4xi8>) -> tensor<1x256x4xi8> {
    %cst = arith.constant dense<[[200, 100, 30, 23, 45, 65, 76, 78]]> : tensor<1x8xi32>
    %cst_0 = arith.constant dense<[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]]> : tensor<1x8x4xi8>
    %0 = tosa.scatter %arg0, %cst, %cst_0 : (tensor<1x256x4xi8>, tensor<1x8xi32>, tensor<1x8x4xi8>) -> tensor<1x256x4xi8>
    return %0 : tensor<1x256x4xi8>
  }
}

