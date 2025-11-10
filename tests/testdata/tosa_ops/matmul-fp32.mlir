module {
  func.func @main(%arg0: tensor<1x128x256xbf16>, %arg1: tensor<1x64x128xbf16>) -> (tensor<1x64x256xf32>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.matmul %arg1, %arg0 : (tensor<1x64x128xbf16>, tensor<1x128x256xbf16>) -> tensor<1x64x256xf32>
    return %0 : tensor<1x64x256xf32>
  }
}
