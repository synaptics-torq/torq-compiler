module {
  func.func @main(%arg0: tensor<1x28x28x512xi8>) -> tensor<1x14x14x512xi8> {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x28x28x512xi8>) -> tensor<1x14x14x512xi8>
    return %0 : tensor<1x14x14x512xi8>
  }
}

