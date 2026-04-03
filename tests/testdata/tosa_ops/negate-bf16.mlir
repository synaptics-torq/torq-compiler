module {
  func.func @main(%arg0: tensor<130x6x3xbf16>) -> tensor<130x6x3xbf16> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %0 = tosa.negate %arg0, %cst, %cst : (tensor<130x6x3xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<130x6x3xbf16>
    return %0 : tensor<130x6x3xbf16>
  }
}

