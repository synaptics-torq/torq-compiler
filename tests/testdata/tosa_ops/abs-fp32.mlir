module {
  func.func @main(%arg0: tensor<1x65x4x3x6xf32>) -> (tensor<1x65x4x3x6xf32>) {
    %0 = "tosa.abs"(%arg0) : (tensor<1x65x4x3x6xf32>) -> tensor<1x65x4x3x6xf32>
    return %0 : tensor<1x65x4x3x6xf32>
  }
}
