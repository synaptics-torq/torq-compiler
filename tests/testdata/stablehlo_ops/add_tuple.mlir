module @jit___call  {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes {} {
    %0 = stablehlo.tuple %arg0, %arg1 : tuple<tensor<i32>, tensor<i32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<i32>, tensor<i32>>) -> tensor<i32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<i32>, tensor<i32>>) -> tensor<i32>
    %3 = stablehlo.add %1, %2 : tensor<i32>
    return %3 : tensor<i32>
  }
}
