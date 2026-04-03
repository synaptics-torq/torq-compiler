module {
  func.func @main(%arg0: tensor<1x2000xf32>, %arg1: tensor<1x2000xf32>) -> tensor<1x2000xf32> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<1x2000xf32>, tensor<1x2000xf32>) -> tensor<1x2000xf32>
    return %0 : tensor<1x2000xf32>
  }
}

