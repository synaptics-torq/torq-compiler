module {
  func.func @main(%arg0: tensor<1300xf32>) -> (tensor<1300xf32>) {
    %0 = tosa.negate %arg0 : (tensor<1300xf32>) -> tensor<1300xf32>
    return %0 : tensor<1300xf32>
  }
}
