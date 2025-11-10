module {
  func.func @main(%arg0: tensor<1x25088xi8>) -> tensor<1x7x7x512xi8> {
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3]] output_shape [1, 7, 7, 512] : tensor<1x25088xi8> into tensor<1x7x7x512xi8>
    return %expanded : tensor<1x7x7x512xi8>
  }
}
