module {
  func.func @main() -> tensor<1x8xi8> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<50> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
    %mult = arith.constant dense<1073741824> : tensor<1xi32>
    %shift = arith.constant dense<30> : tensor<1xi8>
    %izp = arith.constant dense<0> : tensor<1xi32>
    %ozp = arith.constant dense<7> : tensor<1xi8>
    %1 = tosa.rescale %0, %mult, %shift, %izp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8xi8>
    return %1 : tensor<1x8xi8>
  }
}
