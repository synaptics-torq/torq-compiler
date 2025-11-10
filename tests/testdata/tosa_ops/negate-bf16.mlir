module {
  func.func @main(%arg0: tensor<130x6x3xbf16>) -> (tensor<130x6x3xbf16>) {
    %0 = tosa.negate %arg0 : (tensor<130x6x3xbf16>) -> tensor<130x6x3xbf16>
    return %0 : tensor<130x6x3xbf16>
  }
}
