module {
  func.func @main(%arg0: tensor<1x320x320x1xi8>) -> tensor<1x320x320x1xi8> {
    %cst = arith.constant dense<127> : tensor<1xi8>
    %cst_0 = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_1 = arith.constant dense<11> : tensor<1xi8>
    %cst_2 = arith.constant dense<-79> : tensor<1xi8>
    %cst_3 = arith.constant dense<0> : tensor<1xi32>
    %0 = tosa.rescale %arg0, %cst_0, %cst_1, %cst_2, %cst_3 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x320x320x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x320x320x1xi32>
    %cst_4 = arith.constant dense<10> : tensor<1xi8>
    %cst_5 = arith.constant dense<-128> : tensor<1xi8>
    %1 = tosa.rescale %cst, %cst_0, %cst_4, %cst_5, %cst_3 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1xi32>
    %cst_6 = arith.constant dense<1651596561> : tensor<1xi32>
    %cst_7 = arith.constant dense<34> : tensor<1xi8>
    %cst_8 = arith.constant dense<0> : tensor<1xi32>
    %cst_9 = arith.constant dense<0> : tensor<1xi32>
    %2 = tosa.rescale %1, %cst_6, %cst_7, %cst_8, %cst_9 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %3 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.reshape %2, %3 : (tensor<1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
    %5 = tosa.sub %0, %4 : (tensor<1x320x320x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x320x320x1xi32>
    %cst_10 = arith.constant dense<1329334528> : tensor<1xi32>
    %cst_11 = arith.constant dense<49> : tensor<1xi8>
    %cst_12 = arith.constant dense<-67> : tensor<1xi8>
    %6 = tosa.rescale %5, %cst_10, %cst_11, %cst_8, %cst_12 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x320x320x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x320x320x1xi8>
    return %6 : tensor<1x320x320x1xi8>
  }
}

