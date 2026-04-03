module {
  func.func @main(%arg0: tensor<1300xf32>) -> tensor<1300xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %0 = tosa.negate %arg0, %cst, %cst : (tensor<1300xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1300xf32>
    return %0 : tensor<1300xf32>
  }
}

