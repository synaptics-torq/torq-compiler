module {
  func.func @main(%arg0: tensor<1x6xi8>, %arg1: tensor<6xi8>) -> (tensor<1xi16>) {
    %init = tensor.empty() : tensor<1xi16>
    %0 = linalg.matvec ins(%arg0, %arg1 : tensor<1x6xi8>, tensor<6xi8>) outs(%init : tensor<1xi16>) -> tensor<1xi16>
    return %0 : tensor<1xi16>
  }
}
