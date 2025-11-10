module {
  func.func @main(%arg0: tensor<1x320x240x24xi8>, %arg1: tensor<1x320x240x24xi8>) -> (tensor<1x320x240x24xi16>) {
    %2 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<1x320x240x24xi8>, tensor<1x320x240x24xi8>) -> tensor<1x320x240x24xi16>
    return %2 : tensor<1x320x240x24xi16>
  }
}
