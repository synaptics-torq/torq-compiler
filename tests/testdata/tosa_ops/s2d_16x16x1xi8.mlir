module {
  func.func @main(%arg0: tensor<1x16x16x1xi8>) -> (tensor<1x8x8x4xi8>) {
    %0 = "tosa.const"() <{value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}> : () -> tensor<6xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 8, 2, 8, 2, 1>} : (tensor<1x16x16x1xi8>) -> tensor<1x8x2x8x2x1xi8>
    %2 = tosa.transpose %1, %0 : (tensor<1x8x2x8x2x1xi8>, tensor<6xi32>) -> tensor<1x8x8x2x2x1xi8>
    %3 = tosa.reshape %2 {new_shape = array<i64: 1, 8, 8, 4>} : (tensor<1x8x8x2x2x1xi8>) -> tensor<1x8x8x4xi8>
    return %3 : tensor<1x8x8x4xi8>
  }
}

