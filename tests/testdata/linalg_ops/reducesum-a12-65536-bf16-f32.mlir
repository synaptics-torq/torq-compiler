module {
  func.func @main(%a: tensor<4x256x256xbf16>) -> tensor<4xf32> {
    %c = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<4xf32>
    %f = linalg.fill ins(%c : f32) outs(%e : tensor<4xf32>) -> tensor<4xf32>
    %r = linalg.reduce ins(%a : tensor<4x256x256xbf16>) outs(%f : tensor<4xf32>) dimensions = [1, 2]
      (%in: bf16, %init: f32) {
        %x = arith.extf %in : bf16 to f32
        %s = arith.addf %x, %init : f32
        linalg.yield %s : f32
      }
    return %r : tensor<4xf32>
  }
}
