module {
  func.func @main(%arg0: tensor<1x8x8x4xi8>) -> tensor<1x16x16x1xi8> attributes {tf.entry_function = {inputs = "input", outputs = "Identity"}} {
    %0 = tosa.const_shape  {values = dense<[1, 8, 8, 2, 2, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %1 = tosa.const_shape  {values = dense<[1, 16, 16, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %arg0, %0 : (tensor<1x8x8x4xi8>, !tosa.shape<6>) -> tensor<1x8x8x2x2x1xi8>
    %3 = tosa.transpose %2 {perms = array<i32: 0, 1, 3, 2, 4, 5>} : (tensor<1x8x8x2x2x1xi8>) -> tensor<1x8x2x8x2x1xi8>
    %4 = tosa.reshape %3, %1 : (tensor<1x8x2x8x2x1xi8>, !tosa.shape<4>) -> tensor<1x16x16x1xi8>
    return %4 : tensor<1x16x16x1xi8>
  }
}

