module {
  func.func @main(%arg0: tensor<1x1001xi8>) -> tensor<1x1001xui8> {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<-70> : tensor<1xi8>
    %cst_2 = arith.constant dense<58> : tensor<1xi8>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = true, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x1001xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1001xui8>
    return %0 : tensor<1x1001xui8>
  }
}

