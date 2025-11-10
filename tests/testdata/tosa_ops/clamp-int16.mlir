module {
  func.func @main(%arg0: tensor<15x4xi16>) -> (tensor<15x4xi16>) {
    %0 = tosa.clamp %arg0 {min_int = -1270 : i64, max_int = 1260 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<15x4xi16>) -> tensor<15x4xi16>
    return %0 : tensor<15x4xi16>
  }
}
