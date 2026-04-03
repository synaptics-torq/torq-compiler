module {
  func.func @main(%arg0: tensor<1x21x1024xi32>) -> tensor<1x21x1024xi16> {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<0> : tensor<1xi32>
    %cst_2 = arith.constant dense<0> : tensor<1xi16>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x21x1024xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x21x1024xi16>
    return %0 : tensor<1x21x1024xi16>
  }
}

