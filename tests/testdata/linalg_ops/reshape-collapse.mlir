module {
  func.func @main(%arg0: tensor<1x7x7x512xi8>) -> tensor<1x25088xi8> {
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<1x7x7x512xi8> into tensor<1x25088xi8>
    return %collapsed : tensor<1x25088xi8>
  }
}
