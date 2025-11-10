module {
  func.func @main(%arg0: tensor<1x1024x64xi1>) -> (tensor<1x1024x64xi16>) {
    %0 = tosa.cast %arg0 : (tensor<1x1024x64xi1>) -> tensor<1x1024x64xi16>
    return %0 : tensor<1x1024x64xi16>
  }
}
