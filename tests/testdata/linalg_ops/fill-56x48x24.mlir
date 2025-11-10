module {
  func.func @main() -> (tensor<1x56x48x24xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tensor.empty() : tensor<1x56x48x24xi8>
    %cst = arith.constant 5 : i8
    %1 = linalg.fill ins(%cst : i8) outs(%0 : tensor<1x56x48x24xi8>) -> tensor<1x56x48x24xi8>
    return %1 : tensor<1x56x48x24xi8>
  }
}


