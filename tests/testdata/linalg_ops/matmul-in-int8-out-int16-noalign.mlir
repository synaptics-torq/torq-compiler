module {
  func.func @main(%arg0: tensor<12x56xi8>, %arg1: tensor<56x168xi8>) -> (tensor<12x168xi16>) {
    %init = tensor.empty() : tensor<12x168xi16>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<12x56xi8>, tensor<56x168xi8>) outs(%init : tensor<12x168xi16>) -> tensor<12x168xi16>
    return %0 : tensor<12x168xi16>
  }
}

