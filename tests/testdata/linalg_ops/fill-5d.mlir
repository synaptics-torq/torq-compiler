module {
  func.func @main() -> (tensor<3x15x15x2x4xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tensor.empty() : tensor<3x15x15x2x4xi8>
    %cst = arith.constant 5 : i8
    %1 = linalg.fill ins(%cst : i8) outs(%0 : tensor<3x15x15x2x4xi8>) -> tensor<3x15x15x2x4xi8>
    return %1 : tensor<3x15x15x2x4xi8>
  }
}


