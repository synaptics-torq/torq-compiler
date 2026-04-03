module {
  func.func @main(%arg0: tensor<1x8x8x4xi8>) -> tensor<1x9x9x4xi8> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = tosa.const_shape  {values = dense<[0, 0, 1, 0, 1, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
    %2 = tosa.pad %arg0, %1, %0 : (tensor<1x8x8x4xi8>, !tosa.shape<8>, tensor<1xi8>) -> tensor<1x9x9x4xi8>
    return %2 : tensor<1x9x9x4xi8>
  }
}

