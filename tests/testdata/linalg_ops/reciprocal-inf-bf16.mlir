module {
  func.func @main(%arg0: tensor<1x1024x64xbf16>) -> (tensor<1x1024x64xbf16>) {
    %0 = tensor.empty() : tensor<1x1024x64xbf16>
    %1 = linalg.reciprocal ins(%arg0: tensor<1x1024x64xbf16>) outs(%0: tensor<1x1024x64xbf16>) -> tensor<1x1024x64xbf16>
    return %1 : tensor<1x1024x64xbf16>
  }
}
