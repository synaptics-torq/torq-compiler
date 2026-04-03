module {
  func.func @main(%arg0: tensor<1x512x4xi8>, %arg1: tensor<1x256x4xi8>, %arg2: tensor<1x96x4xi8>) -> tensor<1x864x4xi8> {
    %0 = tosa.concat %arg0, %arg1, %arg2 {axis = 1 : i32} : (tensor<1x512x4xi8>, tensor<1x256x4xi8>, tensor<1x96x4xi8>) -> tensor<1x864x4xi8>
    return %0 : tensor<1x864x4xi8>
  }
}

