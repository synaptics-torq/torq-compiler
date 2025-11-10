module {
  func.func @main(%arg0: tensor<1x56x56x24xi32>, %arg1: tensor<1x56x56x24xi32>) -> (tensor<1x56x56x24xi32>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %136 = tosa.sub %arg0, %arg1 : (tensor<1x56x56x24xi32>, tensor<1x56x56x24xi32>) -> tensor<1x56x56x24xi32>
    return %136 : tensor<1x56x56x24xi32>
  }
}

