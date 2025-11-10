module {
  func.func @main(%arg0: tensor<68xi32>) -> (tensor<68xi32>) {
    %0 = tosa.clz %arg0 : (tensor<68xi32>) -> tensor<68xi32>
    return %0 : tensor<68xi32>
  }
}
