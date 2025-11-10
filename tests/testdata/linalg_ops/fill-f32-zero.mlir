// TORQ_ALLOW_ALL_ZERO: 1
module {
  func.func @main() -> (tensor<1x56x56x24xf32>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tensor.empty() : tensor<1x56x56x24xf32>
    %cst = arith.constant 0.0 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
    return %1 : tensor<1x56x56x24xf32>
  }
}
