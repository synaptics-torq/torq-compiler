module {
  func.func @main(%arg0: tensor<1x68x68x7xi8>, %arg1: tensor<1x68x68x7xi8>) -> tensor<1x68x68x7xi32> {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<-128> : tensor<1xi8>
    %cst_2 = arith.constant dense<-128> : tensor<1xi8>
    %cst_3 = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.rescale %arg1, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x68x68x7xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x68x68x7xi8>
    %1 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x68x68x7xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x68x68x7xi8>
    %2 = tosa.mul %0, %1, %cst_3 : (tensor<1x68x68x7xi8>, tensor<1x68x68x7xi8>, tensor<1xi8>) -> tensor<1x68x68x7xi32>
    return %2 : tensor<1x68x68x7xi32>
  }
}

