module {
  func.func @main() -> (tensor<1x56x56x24xbf16>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tensor.empty() : tensor<1x56x56x24xbf16>
    %cst = arith.constant 3.0 : bf16
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<1x56x56x24xbf16>) -> tensor<1x56x56x24xbf16>
    return %1 : tensor<1x56x56x24xbf16>
  }
}
