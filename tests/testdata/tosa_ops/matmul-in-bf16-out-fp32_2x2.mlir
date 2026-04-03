module {
  func.func @main(%arg0: tensor<1x2x1xbf16> {ml_program.identifier = "serving_default_keras_tensor_1:0", tf_saved_model.index_path = ["keras_tensor_1"]}, %arg1: tensor<1x1x2xbf16> {ml_program.identifier = "serving_default_keras_tensor:0", tf_saved_model.index_path = ["keras_tensor"]}) -> (tensor<1x2x2xf32> {ml_program.identifier = "PartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %0 = tosa.matmul %arg0, %arg1, %cst, %cst_0 : (tensor<1x2x1xbf16>, tensor<1x1x2xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x2x2xf32>
    return %0 : tensor<1x2x2xf32>
  }
}

