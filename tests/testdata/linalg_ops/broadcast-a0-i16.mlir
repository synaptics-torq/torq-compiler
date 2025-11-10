module {
  func.func @main(%arg0: tensor<6xi16>) -> (tensor<7x6xi16>) {
    %init = tensor.empty() : tensor<7x6xi16>
    %0 = linalg.broadcast ins(%arg0: tensor<6xi16>) outs(%init: tensor<7x6xi16>) dimensions = [0]
    return %0 : tensor<7x6xi16>
  }
}
