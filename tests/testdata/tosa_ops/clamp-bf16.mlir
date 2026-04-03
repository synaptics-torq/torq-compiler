module {
  func.func @main(%arg0: tensor<1x6x34x23x4x7xbf16>) -> tensor<1x6x34x23x4x7xbf16> {
    %0 = tosa.clamp %arg0 {max_val = 6.000000e+00 : bf16, min_val = 0.000000e+00 : bf16} : (tensor<1x6x34x23x4x7xbf16>) -> tensor<1x6x34x23x4x7xbf16>
    return %0 : tensor<1x6x34x23x4x7xbf16>
  }
}

