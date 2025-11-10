module {
  func.func @main() -> (tensor<1x56x56x24xi32>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tensor.empty() : tensor<1x56x56x24xi32>
    %cst = arith.constant -2147483648 : i32
    %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<1x56x56x24xi32>) -> tensor<1x56x56x24xi32>
    return %1 : tensor<1x56x56x24xi32>
  }
}


