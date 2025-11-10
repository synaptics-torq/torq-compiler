module {
  func.func @main(%arg0: tensor<1x1x1x30xi8>) -> (tensor<1x1x1x30xi1>) {
    %1 = arith.trunci %arg0 : tensor<1x1x1x30xi8> to tensor<1x1x1x30xi1>
    return %1 : tensor<1x1x1x30xi1>
  }
}