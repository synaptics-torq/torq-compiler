module {
  func.func @main(%arg0: tensor<1x56x56x256xi8>, %arg1: tensor<1x56x56x256xi8>) -> tensor<1x56x56x256xi8> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<10> : tensor<1xi8>
    %cst_1 = arith.constant dense<-68> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi32>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x56x56x256xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x56x56x256xi32>
    %cst_3 = arith.constant dense<1385091423> : tensor<1xi32>
    %cst_4 = arith.constant dense<32> : tensor<1xi8>
    %cst_5 = arith.constant dense<0> : tensor<1xi32>
    %1 = tosa.rescale %0, %cst_3, %cst_4, %cst_5, %cst_5 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x56x56x256xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x56x56x256xi32>
    %cst_6 = arith.constant dense<11> : tensor<1xi8>
    %cst_7 = arith.constant dense<11> : tensor<1xi8>
    %2 = tosa.rescale %arg1, %cst, %cst_6, %cst_7, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x56x56x256xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x56x56x256xi32>
    %3 = tosa.sub %1, %2 : (tensor<1x56x56x256xi32>, tensor<1x56x56x256xi32>) -> tensor<1x56x56x256xi32>
    %cst_8 = arith.constant dense<1086272925> : tensor<1xi32>
    %cst_9 = arith.constant dense<48> : tensor<1xi8>
    %cst_10 = arith.constant dense<-128> : tensor<1xi8>
    %4 = tosa.rescale %3, %cst_8, %cst_9, %cst_5, %cst_10 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x56x56x256xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x56x56x256xi8>
    return %4 : tensor<1x56x56x256xi8>
  }
}

