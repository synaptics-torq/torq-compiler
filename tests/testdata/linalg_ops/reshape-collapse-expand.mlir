module {
  func.func @main(%arg0: tensor<1x7x7x512xi8>) -> tensor<1x1x1x25088xi8> {
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<1x7x7x512xi8> into tensor<1x25088xi8>
    %expanded = tensor.expand_shape %collapsed [[0, 1, 2], [3]] output_shape [1, 1, 1, 25088] : tensor<1x25088xi8> into tensor<1x1x1x25088xi8>
    return %expanded : tensor<1x1x1x25088xi8>
  }
}
