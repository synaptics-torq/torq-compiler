module @jit___call  {
  func.func @main(%arg0: tensor<262144x640xbf16>, %6: tensor<1x256x1xi32>) -> tensor<1x256x640xbf16> attributes {} {
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 640>}> : (tensor<262144x640xbf16>, tensor<1x256x1xi32>) -> tensor<1x256x640xbf16>
    return %7 : tensor<1x256x640xbf16>
  }
}
