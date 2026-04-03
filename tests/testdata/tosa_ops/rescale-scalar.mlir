module {
  func.func @main() -> tensor<i32> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<127> : tensor<i8>}> : () -> tensor<i8>
    %cst = arith.constant dense<1073741824> : tensor<1xi32>
    %cst_0 = arith.constant dense<30> : tensor<1xi8>
    %cst_1 = arith.constant dense<-128> : tensor<1xi8>
    %cst_2 = arith.constant dense<0> : tensor<1xi32>
    %1 = tosa.rescale %0, %cst, %cst_0, %cst_1, %cst_2 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<i8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<i32>
    return %1 : tensor<i32>
  }
}

