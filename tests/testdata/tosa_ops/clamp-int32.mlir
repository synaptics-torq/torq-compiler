module {
  func.func @main(%arg0: tensor<1x15x4x3xi32>) -> (tensor<1x15x4x3xi32>) {
    %0 = tosa.clamp %arg0 {min_int = -1270 : i64, max_int = 1260 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1x15x4x3xi32>) -> tensor<1x15x4x3xi32>
    return %0 : tensor<1x15x4x3xi32>
  }
}
