module {
  func.func @main(%arg0: tensor<1x8x207x32xbf16>, %arg1: tensor<1x1x207x32xbf16>) -> tensor<1x8x207x32xbf16> {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.mul %arg0, %arg1, %cst : (tensor<1x8x207x32xbf16>, tensor<1x1x207x32xbf16>, tensor<1xi8>) -> tensor<1x8x207x32xbf16>
    return %0 : tensor<1x8x207x32xbf16>
  }
}

