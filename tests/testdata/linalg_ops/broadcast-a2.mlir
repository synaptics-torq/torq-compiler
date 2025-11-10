module {
  func.func @main(%arg0: tensor<1x21xi8>) -> (tensor<1x21x1024xi8>) {
    %init = tensor.empty() : tensor<1x21x1024xi8>
    %0 = linalg.broadcast ins(%arg0: tensor<1x21xi8>) outs(%init: tensor<1x21x1024xi8>) dimensions = [2]
    return %0 : tensor<1x21x1024xi8>
  }
}
