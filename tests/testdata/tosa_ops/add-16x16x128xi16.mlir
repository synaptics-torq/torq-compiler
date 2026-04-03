module {
  func.func @main(%arg0: tensor<1x16x16x128xi16>, %arg1: tensor<1x16x16x128xi16>) -> tensor<1x16x16x128xi16> {
    %0 = "tosa.const"() <{values = dense<15> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<16> : tensor<1xi8>
    %cst_1 = arith.constant dense<0> : tensor<1xi16>
    %cst_2 = arith.constant dense<0> : tensor<1xi32>
    %1 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x16x16x128xi16>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<1x16x16x128xi32>
    %2 = tosa.cast %arg1 : (tensor<1x16x16x128xi16>) -> tensor<1x16x16x128xi32>
    %3 = tosa.logical_left_shift %2, %0 : (tensor<1x16x16x128xi32>, tensor<1x1x1x1xi32>) -> tensor<1x16x16x128xi32>
    %cst_3 = arith.constant dense<1469854336> : tensor<1xi32>
    %cst_4 = arith.constant dense<42> : tensor<1xi8>
    %cst_5 = arith.constant dense<0> : tensor<1xi32>
    %cst_6 = arith.constant dense<0> : tensor<1xi32>
    %4 = tosa.rescale %3, %cst_3, %cst_4, %cst_5, %cst_6 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x16x16x128xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x16x16x128xi32>
    %5 = tosa.add %1, %4 : (tensor<1x16x16x128xi32>, tensor<1x16x16x128xi32>) -> tensor<1x16x16x128xi32>
    %cst_7 = arith.constant dense<1074460101> : tensor<1xi32>
    %cst_8 = arith.constant dense<44> : tensor<1xi8>
    %cst_9 = arith.constant dense<0> : tensor<1xi32>
    %cst_10 = arith.constant dense<0> : tensor<1xi16>
    %6 = tosa.rescale %5, %cst_7, %cst_8, %cst_9, %cst_10 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x16x16x128xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x16x16x128xi16>
    return %6 : tensor<1x16x16x128xi16>
  }
}

