module {
  func.func @main(%arg0: tensor<1x2x2100xi32>) -> (tensor<1x17x2x2100xi32>) {
    %init = tensor.empty() : tensor<1x17x2x2100xi32>
    %0 = linalg.broadcast ins(%arg0: tensor<1x2x2100xi32>) outs(%init: tensor<1x17x2x2100xi32>) dimensions = [1]
    return %0 : tensor<1x17x2x2100xi32>
  }
}
