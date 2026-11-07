module {
  func.func @main(%arg0: tensor<1x10x10x51xi32>) -> tensor<1x10x10x51xi8> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<1919293440> : tensor<1xi32>
    %cst_0 = arith.constant dense<38> : tensor<1xi8>
    %cst_1 = arith.constant dense<0> : tensor<1xi32>
    %cst_2 = arith.constant dense<-118> : tensor<1xi8>
    %0 = tosa.rescale %arg0, %cst, %cst_0, %cst_1, %cst_2 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x10x10x51xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x10x10x51xi8>
    return %0 : tensor<1x10x10x51xi8>
  }
}

