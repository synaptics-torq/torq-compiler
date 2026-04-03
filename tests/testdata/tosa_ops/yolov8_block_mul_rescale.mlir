module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x160x160x16xi8> {ml_program.identifier = "serving_default_images:0", tf_saved_model.index_path = ["images"]}) -> (tensor<1x160x160x16xi8> {ml_program.identifier = "PartitionedCall:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<5> : tensor<1xi8>
    %cst_2 = arith.constant dense<-128> : tensor<1xi8>
    %cst_3 = arith.constant dense<0> : tensor<1xi32>
    %cst_4 = arith.constant dense<0> : tensor<1xi8>
    %cst_5 = arith.constant dense<1111634816> : tensor<1xi32>
    %cst_6 = arith.constant dense<37> : tensor<1xi8>
    %cst_7 = arith.constant dense<0> : tensor<1xi32>
    %cst_8 = arith.constant dense<-126> : tensor<1xi8>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_3 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x160x160x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x160x160x16xi32>
    %1 = tosa.rescale %arg0, %cst, %cst_0, %cst_2, %cst_3 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x160x160x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x160x160x16xi32>
    %2 = tosa.mul %0, %1, %cst_4 : (tensor<1x160x160x16xi32>, tensor<1x160x160x16xi32>, tensor<1xi8>) -> tensor<1x160x160x16xi32>
    %3 = tosa.rescale %2, %cst_5, %cst_6, %cst_7, %cst_8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x160x160x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x160x160x16xi8>
    return %3 : tensor<1x160x160x16xi8>
  }
}

