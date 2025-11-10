module {
  func.func @main(%arg0: tensor<1x21xf32>) -> (tensor<1x21x1024xf32>) {
    %init = tensor.empty() : tensor<1x21x1024xf32>
    %0 = linalg.broadcast ins(%arg0: tensor<1x21xf32>) outs(%init: tensor<1x21x1024xf32>) dimensions = [2]
    return %0 : tensor<1x21x1024xf32>
  }
}
