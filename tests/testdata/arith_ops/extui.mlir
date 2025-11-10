module {
  func.func @main(%arg0: tensor<1x1x1x30xi1>) -> (tensor<1x1x1x30xi8>) {
    %1 = arith.extui %arg0 : tensor<1x1x1x30xi1> to tensor<1x1x1x30xi8>
    return %1 : tensor<1x1x1x30xi8>
  }
}