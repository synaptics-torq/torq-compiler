module {
  func.func @main(%arg0: tensor<1x2x256x1xi8>) -> tensor<1x2x256x1xi8> {
    %cst = arith.constant dense<127> : tensor<1xi8>
    %cst_0 = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_1 = arith.constant dense<11> : tensor<1xi8>
    %cst_2 = arith.constant dense<-128> : tensor<1xi8>
    %cst_3 = arith.constant dense<0> : tensor<1xi32>
    %0 = tosa.rescale %cst, %cst_0, %cst_1, %cst_2, %cst_3 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1xi32>
    %1 = tosa.rescale %arg0, %cst_0, %cst_1, %cst_2, %cst_3 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x2x256x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x2x256x1xi32>
    %2 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.reshape %0, %2 : (tensor<1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
    %4 = tosa.sub %3, %1 : (tensor<1x1x1x1xi32>, tensor<1x2x256x1xi32>) -> tensor<1x2x256x1xi32>
    %cst_4 = arith.constant dense<49> : tensor<1xi8>
    %cst_5 = arith.constant dense<0> : tensor<1xi32>
    %cst_6 = arith.constant dense<-128> : tensor<1xi8>
    %5 = tosa.rescale %4, %cst_0, %cst_4, %cst_5, %cst_6 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x2x256x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x2x256x1xi8>
    return %5 : tensor<1x2x256x1xi8>
  }
}

