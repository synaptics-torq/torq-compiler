module {                                                                                                                                                                     
  func.func @main(%arg0: tensor<1x7x7x512xi8>) -> (tensor<1x1x1x25088xi8>) {
    %67 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 1, 25088>} : (tensor<1x7x7x512xi8>) -> tensor<1x1x1x25088xi8>
    return %67 : tensor<1x1x1x25088xi8>
  }
}


