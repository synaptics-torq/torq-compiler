module {
  func.func @main(%arg0: tensor<1x2x3x6x65xbf16>) -> (tensor<1x2x3x6x65xbf16>) {
    %0 = "tosa.abs"(%arg0) : (tensor<1x2x3x6x65xbf16>) -> tensor<1x2x3x6x65xbf16>
    return %0 : tensor<1x2x3x6x65xbf16>
  }
}
