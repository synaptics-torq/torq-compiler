module {
  func.func @main(%arg0: tensor<1x7x7x128xi8>) -> tensor<1x1x1x128xi8> {
    %cst = arith.constant dense<4> : tensor<1xi8>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.avg_pool2d %arg0, %cst, %cst_0 {acc_type = i32, kernel = array<i64: 7, 7>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x128xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x1x128xi8>
    return %0 : tensor<1x1x1x128xi8>
  }
}

