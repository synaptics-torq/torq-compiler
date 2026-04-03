module {
  func.func @main(%arg0: tensor<1x16x128x1xi8> {ml_program.identifier = "model_11/tf.compat.v1.transpose_5/transpose;"}, %arg1: tensor<1x1x1x1xi8> {ml_program.identifier = "model_11/tf.math.reduce_mean/Mean"}) -> (tensor<1x16x128x1xi8> {ml_program.identifier = "model_11/tf.math.subtract/Sub"}) {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<11> : tensor<1xi8>
    %cst_1 = arith.constant dense<6> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi32>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x16x128x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x16x128x1xi32>
    %cst_3 = arith.constant dense<10> : tensor<1xi8>
    %cst_4 = arith.constant dense<-128> : tensor<1xi8>
    %1 = tosa.rescale %arg1, %cst, %cst_3, %cst_4, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x1x1xi32>
    %cst_5 = arith.constant dense<2039850423> : tensor<1xi32>
    %cst_6 = arith.constant dense<39> : tensor<1xi8>
    %cst_7 = arith.constant dense<0> : tensor<1xi32>
    %2 = tosa.rescale %1, %cst_5, %cst_6, %cst_7, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x1x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x1x1xi32>
    %3 = tosa.sub %0, %2 : (tensor<1x16x128x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x16x128x1xi32>
    %cst_8 = arith.constant dense<1077948431> : tensor<1xi32>
    %cst_9 = arith.constant dense<49> : tensor<1xi8>
    %cst_10 = arith.constant dense<0> : tensor<1xi32>
    %cst_11 = arith.constant dense<7> : tensor<1xi8>
    %4 = tosa.rescale %3, %cst_8, %cst_9, %cst_10, %cst_11 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x16x128x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x16x128x1xi8>
    return %4 : tensor<1x16x128x1xi8>
  }
}

