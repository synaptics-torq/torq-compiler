module {
  func.func @main(%a: tensor<16x32x128xi32>) -> tensor<16xi32> {
    %c = arith.constant 0 : i32
    %e = tensor.empty() : tensor<16xi32>
    %f = linalg.fill ins(%c : i32) outs(%e : tensor<16xi32>) -> tensor<16xi32>
    %r = linalg.reduce ins(%a : tensor<16x32x128xi32>) outs(%f : tensor<16xi32>) dimensions = [1, 2]
      (%in: i32, %init: i32) {
        %s = arith.addi %in, %init : i32
        linalg.yield %s : i32
      }
    return %r : tensor<16xi32>
  }
}
