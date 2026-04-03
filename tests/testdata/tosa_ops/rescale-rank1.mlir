module {
  func.func @main(%arg0: tensor<1x17x2x2100xi8>) -> tensor<1x17x2x2100xi32> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<11> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi32>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x17x2x2100xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x17x2x2100xi32>
    return %0 : tensor<1x17x2x2100xi32>
  }
}

