module {
  func.func @main(%arg0: tensor<1x256x16xi8>) -> (tensor<1x256x16xbf16>) {
    %0 = tosa.cast %arg0 : (tensor<1x256x16xi8>) -> tensor<1x256x16xbf16>
    return %0 : tensor<1x256x16xbf16>
  }
}
