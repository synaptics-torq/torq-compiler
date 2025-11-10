module {
  func.func @main(%arg0: tensor<1x68x68x7xbf16>, %arg1: tensor<1x68x68x7xbf16>) -> (tensor<1x68x68x7xbf16>) {
    %2 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<1x68x68x7xbf16>, tensor<1x68x68x7xbf16>) -> tensor<1x68x68x7xbf16>
    return %2 : tensor<1x68x68x7xbf16>
  }
}
