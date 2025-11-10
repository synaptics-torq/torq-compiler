module {
  func.func @main(%arg0: tensor<1x1024x12x8xbf16>) -> (tensor<1x1024x12xbf16>) {
    %cst = arith.constant 1.0 : bf16
    %0 = tensor.empty() : tensor<1x1024x12xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<1x1024x12xbf16>) -> tensor<1x1024x12xbf16>
    %reduced = linalg.reduce ins(%arg0 : tensor<1x1024x12x8xbf16>) outs(%1 : tensor<1x1024x12xbf16>) dimensions = [3] 
      (%in: bf16, %init: bf16) {
        %139 = arith.mulf %in, %init : bf16
        linalg.yield %139 : bf16
      }
    return %reduced : tensor<1x1024x12xbf16>
  }
}