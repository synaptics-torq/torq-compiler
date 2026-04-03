module {
  func.func @main(%arg0: tensor<1x128x256xbf16>, %arg1: tensor<1x64x128xbf16>) -> tensor<1x64x256xf32> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %0 = tosa.matmul %arg1, %arg0, %cst, %cst_0 : (tensor<1x64x128xbf16>, tensor<1x128x256xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x64x256xf32>
    return %0 : tensor<1x64x256xf32>
  }
}

