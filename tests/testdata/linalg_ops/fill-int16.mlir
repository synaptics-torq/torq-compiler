module {
  func.func @main() -> (tensor<1x56x56x24xi16>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tensor.empty() : tensor<1x56x56x24xi16>
    %cst = arith.constant 256 : i16
    %1 = linalg.fill ins(%cst : i16) outs(%0 : tensor<1x56x56x24xi16>) -> tensor<1x56x56x24xi16>
    return %1 : tensor<1x56x56x24xi16>
  }
}


