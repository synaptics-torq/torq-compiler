module {
  func.func @main(%arg0: tensor<6xi8>) -> (tensor<7x6xi8>) {
    %init = tensor.empty() : tensor<7x6xi8>
    %0 = linalg.broadcast ins(%arg0: tensor<6xi8>) outs(%init: tensor<7x6xi8>) dimensions = [0]
    return %0 : tensor<7x6xi8>
  }
}
