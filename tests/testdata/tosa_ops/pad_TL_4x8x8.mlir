module {
  func.func @main(%arg0: tensor<1x8x8x4xi8>) -> tensor<1x9x9x4xi8> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x8x8x4xi8>) -> tensor<1x4x8x8xi8>
    %cst = arith.constant dense<19> : tensor<1xi8>
    %1 = tosa.const_shape  {values = dense<[0, 0, 0, 0, 1, 0, 1, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
    %2 = tosa.pad %0, %1, %cst : (tensor<1x4x8x8xi8>, !tosa.shape<8>, tensor<1xi8>) -> tensor<1x4x9x9xi8>
    %3 = tosa.transpose %2 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x4x9x9xi8>) -> tensor<1x9x9x4xi8>
    return %3 : tensor<1x9x9x4xi8>
  }
}

