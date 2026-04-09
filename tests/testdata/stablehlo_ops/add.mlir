module @jit___call  {
  func.func @main(%2: tensor<i32>, %3: tensor<i32>) -> tensor<i32> attributes {} {
    %9 = stablehlo.add %3, %2 : tensor<i32>
    return %9 : tensor<i32>
  }
}
