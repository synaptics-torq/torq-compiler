module {
  func.func @main(%arg0: tensor<1x56x56x24xi8> {ml_program.identifier = "serving_default_input_1:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x56x56x24xi8> {ml_program.identifier = "mobilenetv2_1.00_224/predictions/MatMul;mobilenetv2_1.00_224/predictions/BiasAdd", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<11> : tensor<1xi8>
    %cst_1 = arith.constant dense<-3> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi32>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x56x56x24xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x56x56x24xi32>
    %1 = tosa.add %0, %0 : (tensor<1x56x56x24xi32>, tensor<1x56x56x24xi32>) -> tensor<1x56x56x24xi32>
    %cst_3 = arith.constant dense<1093202297> : tensor<1xi32>
    %cst_4 = arith.constant dense<49> : tensor<1xi8>
    %cst_5 = arith.constant dense<0> : tensor<1xi32>
    %cst_6 = arith.constant dense<-2> : tensor<1xi8>
    %2 = tosa.rescale %1, %cst_3, %cst_4, %cst_5, %cst_6 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x56x56x24xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x56x56x24xi8>
    return %2 : tensor<1x56x56x24xi8>
  }
}

