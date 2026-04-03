module {
  func.func @main(%arg0: tensor<1xi8>) -> tensor<1xi8> {
    %0 = tosa.clamp %arg0 {max_val = 126 : i8, min_val = -127 : i8} : (tensor<1xi8>) -> tensor<1xi8>
    return %0 : tensor<1xi8>
  }
}

