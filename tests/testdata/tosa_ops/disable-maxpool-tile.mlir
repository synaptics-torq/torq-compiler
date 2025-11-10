module {
  func.func @main(%arg0: tensor<1x114x114x64xi8>) -> (tensor<1x56x56x64xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %119 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x114x114x64xi8>) -> tensor<1x56x56x64xi8>
    return %119 : tensor<1x56x56x64xi8>
  }
}

