// The dims are not contiguous, so the reduce is a chain.
module {
  func.func @main(%a: tensor<2x8x2xi32>) -> tensor<8xi32> {
    %c = arith.constant 5 : i32
    %e = tensor.empty() : tensor<8xi32>
    %f = linalg.fill ins(%c : i32) outs(%e : tensor<8xi32>) -> tensor<8xi32>
    %r = linalg.reduce ins(%a : tensor<2x8x2xi32>) outs(%f : tensor<8xi32>) dimensions = [0, 2]
      (%in: i32, %init: i32) {
        %s = arith.addi %in, %init : i32
        linalg.yield %s : i32
      }
    return %r : tensor<8xi32>
  }
}
