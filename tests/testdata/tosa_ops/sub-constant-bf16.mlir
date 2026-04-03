module {
  func.func @main(%arg0: tensor<1x21x1024xbf16>) -> tensor<1x21x1024xbf16> {
    %0 = "tosa.const"() <{values = dense<3.276800e+04> : tensor<1x1x1xbf16>}> : () -> tensor<1x1x1xbf16>
    %1 = tosa.sub %arg0, %0 : (tensor<1x21x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<1x21x1024xbf16>
    return %1 : tensor<1x21x1024xbf16>
  }
}

