module {
  func.func @main(%arg0: tensor<6xi8>) -> (tensor<6x7x3xi8>) {
    %init = tensor.empty() : tensor<6x7x3xi8>
    %0 = linalg.broadcast ins(%arg0: tensor<6xi8>) outs(%init: tensor<6x7x3xi8>) dimensions = [1,2]
    return %0 : tensor<6x7x3xi8>
  }
}
