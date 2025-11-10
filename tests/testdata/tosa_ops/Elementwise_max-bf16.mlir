module {
  func.func @main(%arg0: tensor<4x64xbf16>, %arg1: tensor<4x64xbf16>) -> tensor<4x64xbf16> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<4x64xbf16>, tensor<4x64xbf16>) -> tensor<4x64xbf16>
    return %0 : tensor<4x64xbf16>
  }
}