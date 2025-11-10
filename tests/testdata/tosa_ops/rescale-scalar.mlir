module {
  func.func @main() -> (tensor<i32>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %9 = "tosa.const"() <{value = dense<127> : tensor<i8>}> : () -> tensor<i8>
    %742 = tosa.rescale %9 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<i8>) -> tensor<i32>
    return %742 : tensor<i32>
  }
}

