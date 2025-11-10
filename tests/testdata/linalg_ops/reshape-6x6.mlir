module {
  func.func @main(%arg0: tensor<6x6xi8>) -> tensor<1x1x6x6xi8> {
    %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [1, 1, 6, 6] : tensor<6x6xi8> into tensor<1x1x6x6xi8>
    return %expanded : tensor<1x1x6x6xi8>
  }
}
