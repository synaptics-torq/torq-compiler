module @jit___call  {
  func.func @main(%0: tensor<1024x1024xi32>, %1: tensor<1024x1024xi32>) -> tensor<1024x1024xi1> attributes {} {
    %2 = stablehlo.compare NE, %0, %1, SIGNED : (tensor<1024x1024xi32>, tensor<1024x1024xi32>) -> tensor<1024x1024xi1>
    return %2 : tensor<1024x1024xi1>
  }
}

