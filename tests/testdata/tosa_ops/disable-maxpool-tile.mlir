module {
  func.func @main(%arg0: tensor<1x114x114x64xi8>) -> tensor<1x56x56x64xi8> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x114x114x64xi8>) -> tensor<1x57x57x64xi8>
    %1 = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.const_shape  {values = dense<[1, 56, 56, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.slice %0, %1, %2 : (tensor<1x57x57x64xi8>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x56x56x64xi8>
    return %3 : tensor<1x56x56x64xi8>
  }
}

