module {
  func.func @main(%arg0: tensor<1x21x1xi16>, %arg1: tensor<1x21x1024xi16>) -> (tensor<1x21x1024xi32>) {
    %2 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<1x21x1xi16>, tensor<1x21x1024xi16>) -> tensor<1x21x1024xi32>
    return %2 : tensor<1x21x1024xi32>
  }
}
