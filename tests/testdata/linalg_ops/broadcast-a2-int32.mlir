module {
  func.func @main(%arg0: tensor<1x21xi32>) -> (tensor<1x21x1024xi32>) {
    %init = tensor.empty() : tensor<1x21x1024xi32>
    %0 = linalg.broadcast ins(%arg0: tensor<1x21xi32>) outs(%init: tensor<1x21x1024xi32>) dimensions = [2]
    return %0 : tensor<1x21x1024xi32>
  }
}
