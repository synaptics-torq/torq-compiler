    func.func @main(%270 : tensor<1x2000xf32>, %271: tensor<1x2000xf32>) -> tensor<1x2000xf32> {
        %272 = tosa.sub %270, %271 : (tensor<1x2000xf32>, tensor<1x2000xf32>) -> tensor<1x2000xf32>
        return %272 : tensor<1x2000xf32>
    }
