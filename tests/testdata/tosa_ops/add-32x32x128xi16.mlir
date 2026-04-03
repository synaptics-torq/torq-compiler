module {
  func.func @main(%arg0: tensor<1x32x32x128xi16>, %arg1: tensor<1x32x32x128xi16>) -> tensor<1x32x32x128xi16> {
    %cst = arith.constant dense<15> : tensor<1x1x1x1xi32>
    %cst_0 = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_1 = arith.constant dense<16> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi16>
    %cst_3 = arith.constant dense<0> : tensor<1xi32>
    %0 = tosa.rescale %arg0, %cst_0, %cst_1, %cst_2, %cst_3 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x32x32x128xi16>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<1x32x32x128xi32>
    %1 = tosa.cast %arg1 : (tensor<1x32x32x128xi16>) -> tensor<1x32x32x128xi32>
    %2 = tosa.logical_left_shift %1, %cst : (tensor<1x32x32x128xi32>, tensor<1x1x1x1xi32>) -> tensor<1x32x32x128xi32>
    %cst_4 = arith.constant dense<1730653413> : tensor<1xi32>
    %cst_5 = arith.constant dense<42> : tensor<1xi8>
    %cst_6 = arith.constant dense<0> : tensor<1xi32>
    %cst_7 = arith.constant dense<0> : tensor<1xi32>
    %3 = tosa.rescale %2, %cst_4, %cst_5, %cst_6, %cst_7 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x32x32x128xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x32x32x128xi32>
    %4 = tosa.add %0, %3 : (tensor<1x32x32x128xi32>, tensor<1x32x32x128xi32>) -> tensor<1x32x32x128xi32>
    %cst_8 = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_9 = arith.constant dense<44> : tensor<1xi8>
    %cst_10 = arith.constant dense<0> : tensor<1xi32>
    %cst_11 = arith.constant dense<0> : tensor<1xi16>
    %5 = tosa.rescale %4, %cst_8, %cst_9, %cst_10, %cst_11 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x32x32x128xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x32x32x128xi16>
    return %5 : tensor<1x32x32x128xi16>
  }
}

