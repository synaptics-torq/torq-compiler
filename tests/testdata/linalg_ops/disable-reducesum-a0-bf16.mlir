module {
  func.func @main(%arg0: tensor<6x64x128xbf16>) -> (tensor<64x128xbf16>) {
    %cst = arith.constant 0.0 : bf16
    %0 = tensor.empty() : tensor<64x128xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %reduced = linalg.reduce ins(%arg0 : tensor<6x64x128xbf16>) outs(%1 : tensor<64x128xbf16>) dimensions = [0] 
      (%in: bf16, %init: bf16) {
        %11 = arith.addf %in, %init : bf16
        linalg.yield %11 : bf16
      }
    return %reduced : tensor<64x128xbf16>
  }
}
