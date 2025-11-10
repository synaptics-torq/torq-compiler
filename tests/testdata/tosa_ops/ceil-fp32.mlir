module {
  func.func @main(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = tosa.ceil %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
}
