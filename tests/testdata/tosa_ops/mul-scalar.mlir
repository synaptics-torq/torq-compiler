module {
  func.func @main(%188: tensor<1x8x207x32xbf16>, %22: tensor<1x1x207x32xbf16>) -> (tensor<1x8x207x32xbf16>) {
    %189 = tosa.mul %188, %22 {shift = 0 : i8} : (tensor<1x8x207x32xbf16>, tensor<1x1x207x32xbf16>) -> tensor<1x8x207x32xbf16>
    return %189 : tensor<1x8x207x32xbf16>
  }
}
