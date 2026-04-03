module {
  func.func @main(%arg0: tensor<1x24x3xf32>) -> tensor<1x24x3xf32> {
    %0 = tosa.clamp %arg0 {max_val = 6.000000e+00 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x24x3xf32>) -> tensor<1x24x3xf32>
    return %0 : tensor<1x24x3xf32>
  }
}

