module {
  func.func @main(%arg0: tensor<1x20x20x16xi8>) -> tensor<1x20x20x64xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 5, 5>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x20x20x16xi8>) -> tensor<1x20x20x16xi8>
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 9, 9>, pad = array<i64: 4, 4, 4, 4>, stride = array<i64: 1, 1>} : (tensor<1x20x20x16xi8>) -> tensor<1x20x20x16xi8>
    %2 = tosa.max_pool2d %arg0 {kernel = array<i64: 13, 13>, pad = array<i64: 6, 6, 6, 6>, stride = array<i64: 1, 1>} : (tensor<1x20x20x16xi8>) -> tensor<1x20x20x16xi8>
    %3 = tosa.concat %arg0, %0, %1, %2 {axis = 3 : i32} : (tensor<1x20x20x16xi8>, tensor<1x20x20x16xi8>, tensor<1x20x20x16xi8>, tensor<1x20x20x16xi8>) -> tensor<1x20x20x64xi8>
    return %3 : tensor<1x20x20x64xi8>
  }
}
