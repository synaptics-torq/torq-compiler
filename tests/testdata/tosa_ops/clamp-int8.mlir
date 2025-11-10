module {
  func.func @main(%arg0: tensor<1xi8>) -> (tensor<1xi8>) {
    %0 = tosa.clamp %arg0 {min_int = -127 : i64, max_int = 126 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1xi8>) -> tensor<1xi8>
    return %0 : tensor<1xi8>
  }
}
