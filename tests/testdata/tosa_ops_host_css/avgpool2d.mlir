module {
  func.func @main(%arg0: tensor<1x7x7x128xi8>) -> tensor<1x1x1x128xi8> {
    %cst = arith.constant dense<4> : tensor<1xi8>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.avg_pool2d %arg0, %cst, %cst_0 kernel([7, 7]) stride([1, 1]) pad([0, 0, 0, 0]) acc_type(i32) : (tensor<1x7x7x128xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x1x128xi8>
    return %0 : tensor<1x1x1x128xi8>
  }
}

