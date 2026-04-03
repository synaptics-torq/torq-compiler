module {
  func.func @main(%arg0: tensor<15x4xi16>) -> tensor<15x4xi16> {
    %0 = tosa.clamp %arg0 {max_val = 1260 : i16, min_val = -1270 : i16} : (tensor<15x4xi16>) -> tensor<15x4xi16>
    return %0 : tensor<15x4xi16>
  }
}

