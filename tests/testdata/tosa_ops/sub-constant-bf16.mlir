module {
  func.func @main(%188: tensor<1x21x1024xbf16>) -> (tensor<1x21x1024xbf16>) {
    %22 = "tosa.const"() <{value = dense<32767.0> : tensor<1x1x1xbf16>}> : () -> tensor<1x1x1xbf16>
    %189 = tosa.sub %188, %22 : (tensor<1x21x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<1x21x1024xbf16>
    return %189 : tensor<1x21x1024xbf16>
  }
}
