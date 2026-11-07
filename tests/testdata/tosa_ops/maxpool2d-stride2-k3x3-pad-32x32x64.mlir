module {
  func.func @main(%arg0: tensor<1x32x32x64xi8>) -> tensor<1x16x16x64xi8> {
    %0 = tosa.const_shape values(dense<[0, 0, 0, 1, 0, 1, 0, 0]> : tensor<8xindex>) : () -> !tosa.shape<8>
    %1 = "tosa.const"() <{values = dense<-100> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = tosa.pad %arg0, %0, %1 : (tensor<1x32x32x64xi8>, !tosa.shape<8>, tensor<1xi8>) -> tensor<1x33x33x64xi8>
    %3 = tosa.max_pool2d %2 kernel([3, 3]) stride([2, 2]) pad([0, 0, 0, 0]) : (tensor<1x33x33x64xi8>) -> tensor<1x16x16x64xi8>
    return %3 : tensor<1x16x16x64xi8>
  }
}

