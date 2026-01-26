module  {
  func.func @main(%19: tensor<1x1024x64xi8>) -> tensor<1x1024x64xi16> attributes {} {
    %20 = arith.extui %19 : tensor<1x1024x64xi8> to tensor<1x1024x64xi16>
    return %20 : tensor<1x1024x64xi16>
  }
}