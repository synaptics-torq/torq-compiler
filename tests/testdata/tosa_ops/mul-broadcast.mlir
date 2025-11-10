module {
  func.func @main(%751: tensor<1x17x2x2100xi16>, %752: tensor<1x1x2x2100xi16>) -> (tensor<1x17x2x2100xi16>) {
    %753 = tosa.mul %751, %752 {shift = 0 : i8} : (tensor<1x17x2x2100xi16>, tensor<1x1x2x2100xi16>) -> tensor<1x17x2x2100xi16>
    return %753 : tensor<1x17x2x2100xi16>
  }
}
