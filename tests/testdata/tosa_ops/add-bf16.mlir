module {
  func.func @main(%arg0: tensor<1x56x56x24xbf16>, %arg1: tensor<1x56x56x24xbf16>) -> (tensor<1x56x56x24xbf16>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %136 = tosa.add %arg0, %arg1 : (tensor<1x56x56x24xbf16>, tensor<1x56x56x24xbf16>) -> tensor<1x56x56x24xbf16>
    return %136 : tensor<1x56x56x24xbf16>
  }
}

