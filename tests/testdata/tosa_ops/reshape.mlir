module {
  func.func @main(%arg0: tensor<1x7x7x512xi8>) -> tensor<1x1x1x25088xi8> {
    %0 = tosa.const_shape  {values = dense<[1, 1, 1, 25088]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %arg0, %0 : (tensor<1x7x7x512xi8>, !tosa.shape<4>) -> tensor<1x1x1x25088xi8>
    return %1 : tensor<1x1x1x25088xi8>
  }
}

