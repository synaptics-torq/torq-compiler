// TORQ_FP_MAX_TOL: 0.0001
module {
  func.func @main(%arg0: tensor<6x64x128xbf16>) -> (tensor<64x128xf32>) {
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<6x64x128xbf16>) outs(%1 : tensor<64x128xf32>) dimensions = [0] 
      (%in: bf16, %init: f32) {
        %ext = arith.extf %in : bf16 to f32
        %11 = arith.addf %ext, %init : f32
        linalg.yield %11 : f32
      }
    return %reduced : tensor<64x128xf32>
  }
}
