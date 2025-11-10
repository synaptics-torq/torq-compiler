module {
  func.func @main(%799: tensor<1x4x2100x16xf32>) -> (tensor<1x4x2100x16xf32>) attributes {tf_saved_model.exported_names = ["serving_default"]} {

    %7 = "tosa.const"() <{value = dense<5.900000e+01> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %800 = tosa.sub %799, %7 : (tensor<1x4x2100x16xf32>, tensor<1x1x1x1xf32>) -> tensor<1x4x2100x16xf32>

    return %800 : tensor<1x4x2100x16xf32>
  }
}
