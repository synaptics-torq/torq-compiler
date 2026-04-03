module {
  func.func @main(%arg0: tensor<1x4x2100x16xf32>) -> tensor<1x4x2100x16xf32> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<5.900000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %1 = tosa.sub %arg0, %0 : (tensor<1x4x2100x16xf32>, tensor<1x1x1x1xf32>) -> tensor<1x4x2100x16xf32>
    return %1 : tensor<1x4x2100x16xf32>
  }
}

