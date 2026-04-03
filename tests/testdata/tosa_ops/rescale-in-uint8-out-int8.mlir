module {
  func.func @main(%arg0: tensor<1x64x64x4xui8>) -> tensor<1x64x64x4xi8> {
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<128> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x64x64x4xui8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x64x64x4xi8>
    return %0 : tensor<1x64x64x4xi8>
  }
}

