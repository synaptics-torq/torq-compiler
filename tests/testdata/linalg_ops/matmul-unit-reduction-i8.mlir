// Near miss for the unit-reduction matmul fold: the element types are integer,
// which the fold's float dtype guard rejects, so this k == 1 / n == 1 i8
// matmul must stay a matmul (a rewrite would emit a float product body).
module {
  func.func @main(%arg0: tensor<9x1xi8>, %arg1: tensor<1x1xi8>) -> (tensor<9x1xi16>) {
    %cst = arith.constant 0 : i16
    %0 = tensor.empty() : tensor<9x1xi16>
    %1 = linalg.fill ins(%cst : i16) outs(%0 : tensor<9x1xi16>) -> tensor<9x1xi16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<9x1xi8>, tensor<1x1xi8>) outs(%1 : tensor<9x1xi16>) -> tensor<9x1xi16>
    return %2 : tensor<9x1xi16>
  }
}
