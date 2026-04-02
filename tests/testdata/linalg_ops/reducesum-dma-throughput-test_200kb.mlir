module {
  func.func @main(%arg0: tensor<200000x1xi8>) -> (tensor<1xi8>) {
    %cst = arith.constant 0 : i8
    %0 = tensor.empty() : tensor<1xi8>
    %1 = linalg.fill ins(%cst : i8) outs(%0 : tensor<1xi8>) -> tensor<1xi8>
    %reduced = linalg.reduce ins(%arg0 : tensor<200000x1xi8>) outs(%1 : tensor<1xi8>) dimensions = [0] 
      (%in: i8, %init: i8) {
        %11 = arith.addi %in, %init : i8
        linalg.yield %11 : i8
      }
    return %reduced : tensor<1xi8>
  }
}
