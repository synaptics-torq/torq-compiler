module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x112x112x32xi8> {ml_program.identifier = "efficientnetb0_1/stem_conv_1/convolution;;efficientnetb0_1/stem_conv_pad_1/Pad1", tf_saved_model.index_path = ["input_164"]}, %arg1: tensor<1x112x112x32xi8> {ml_program.identifier = "efficientnetb0_1/stem_activation_1/Sigmoid", tf_saved_model.index_path = ["input_165"]}) -> (tensor<1x112x112x32xi8> {ml_program.identifier = "efficientnetb0_1/stem_activation_1/mul_1", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<0> : tensor<256xi8>}> : () -> tensor<256xi8>
    %1 = tosa.table %arg1, %0 : (tensor<1x112x112x32xi8>, tensor<256xi8>) -> tensor<1x112x112x32xi8>
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<-26> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi32>
    %2 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x112x112x32xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x112x112x32xi32>
    %cst_3 = arith.constant dense<-128> : tensor<1xi8>
    %3 = tosa.rescale %1, %cst, %cst_0, %cst_3, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x112x112x32xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x112x112x32xi32>
    %cst_4 = arith.constant dense<0> : tensor<1xi8>
    %4 = tosa.mul %2, %3, %cst_4 : (tensor<1x112x112x32xi32>, tensor<1x112x112x32xi32>, tensor<1xi8>) -> tensor<1x112x112x32xi32>
    %cst_5 = arith.constant dense<2075768064> : tensor<1xi32>
    %cst_6 = arith.constant dense<38> : tensor<1xi8>
    %cst_7 = arith.constant dense<0> : tensor<1xi32>
    %cst_8 = arith.constant dense<-26> : tensor<1xi8>
    %5 = tosa.rescale %4, %cst_5, %cst_6, %cst_7, %cst_8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x112x112x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x112x112x32xi8>
    return %5 : tensor<1x112x112x32xi8>
  }
}

