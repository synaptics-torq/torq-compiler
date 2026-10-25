module {
  func.func @main(%arg0: tensor<2x48x7x9xi8>) -> tensor<2x7x9x48xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x48x7x9xi8>) -> tensor<2x7x9x48xi8>
    return %0 : tensor<2x7x9x48xi8>
  }
}
