module {
  func.func @main(%arg0: tensor<1x7x7x1280xi8>) -> tensor<1x1280xi8> {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<-128> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi32>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x7x7x1280xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x7x7x1280xi32>
    %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x7x7x1280xi32>) -> tensor<1x1x7x1280xi32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<1x1x7x1280xi32>) -> tensor<1x1x1x1280xi32>
    %cst_3 = arith.constant dense<730269779> : tensor<1xi32>
    %cst_4 = arith.constant dense<35> : tensor<1xi8>
    %cst_5 = arith.constant dense<0> : tensor<1xi32>
    %cst_6 = arith.constant dense<-128> : tensor<1xi8>
    %3 = tosa.rescale %2, %cst_3, %cst_4, %cst_5, %cst_6 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x1x1280xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x1x1280xi8>
    %4 = tosa.const_shape  {values = dense<[-1, 1280]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %3, %4 : (tensor<1x1x1x1280xi8>, !tosa.shape<2>) -> tensor<1x1280xi8>
    return %5 : tensor<1x1280xi8>
  }
}

