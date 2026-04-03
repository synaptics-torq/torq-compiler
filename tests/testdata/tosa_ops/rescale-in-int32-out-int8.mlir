module {
  func.func @main(%arg0: tensor<1x56x56x3xi32>) -> tensor<1x56x56x3xi8> {
    %cst = arith.constant dense<1106700928> : tensor<1xi32>
    %cst_0 = arith.constant dense<45> : tensor<1xi8>
    %cst_1 = arith.constant dense<0> : tensor<1xi32>
    %cst_2 = arith.constant dense<125> : tensor<1xi8>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x56x56x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x56x56x3xi8>
    return %0 : tensor<1x56x56x3xi8>
  }
}

