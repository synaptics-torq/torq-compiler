module {
  func.func @main(%arg0: tensor<12x56xi8>, %arg1: tensor<56x168xi8>) -> (tensor<12x168xi16>) {
    %cst = arith.constant 0 : i16
    %0 = tensor.empty() : tensor<12x168xi16>
    %1 = linalg.fill ins(%cst : i16) outs(%0 : tensor<12x168xi16>) -> tensor<12x168xi16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<12x56xi8>, tensor<56x168xi8>) outs(%1 : tensor<12x168xi16>) -> tensor<12x168xi16>
    return %2 : tensor<12x168xi16>
  }
}

