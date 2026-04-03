module {
  func.func @main(%arg0: tensor<1x7x7x1280xi8>) -> tensor<1x1280xi8> {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.avg_pool2d %arg0, %cst, %cst {acc_type = i32, kernel = array<i64: 7, 7>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x1280xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x1x1280xi8>
    %1 = tosa.const_shape  {values = dense<[1, 1280]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %0, %1 : (tensor<1x1x1x1280xi8>, !tosa.shape<2>) -> tensor<1x1280xi8>
    return %2 : tensor<1x1280xi8>
  }
}

