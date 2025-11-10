module {
  func.func @main(%arg0: tensor<1x1024x64xbf16>) -> (tensor<1x1024x64xbf16>) {
    %0 = math.erf %arg0 : tensor<1x1024x64xbf16>
    return %0 : tensor<1x1024x64xbf16>
  }
}
