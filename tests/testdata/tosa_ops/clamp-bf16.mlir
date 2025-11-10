module {
  func.func @main(%arg0: tensor<1x6x34x23x4x7xbf16>) -> (tensor<1x6x34x23x4x7xbf16>) {
    %0 = tosa.clamp %arg0 {min_int = 0 : i64, max_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 6.0 : f32} : (tensor<1x6x34x23x4x7xbf16>) -> tensor<1x6x34x23x4x7xbf16>
    return %0 : tensor<1x6x34x23x4x7xbf16>
  }
}
