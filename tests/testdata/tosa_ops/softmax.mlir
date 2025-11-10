    func.func @main(%270 : tensor<1x1000xi8> ) -> tensor<1x1000xi8> {
        %18 = "tosa.const"() <{value = dense<-128> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
        %19 = "tosa.const"() <{value = dense<2.560000e+02> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
        %20 = "tosa.const"() <{value = dense<0.0673685893> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
        %21 = "tosa.const"() <{value = dense<-4.800000e+01> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
        %271 = tosa.cast %270 : (tensor<1x1000xi8>) -> tensor<1x1000xf32>
        %272 = tosa.sub %271, %21 : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
        %273 = tosa.mul %272, %20 {shift = 0 : i8} : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
        %274 = tosa.reduce_max %273 {axis = 1 : i32} : (tensor<1x1000xf32>) -> tensor<1x1xf32>
        %275 = tosa.sub %273, %274 : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
        %276 = tosa.exp %275 : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
        %277 = tosa.reduce_sum %276 {axis = 1 : i32} : (tensor<1x1000xf32>) -> tensor<1x1xf32>
        %278 = tosa.reciprocal %277 : (tensor<1x1xf32>) -> tensor<1x1xf32>
        %279 = tosa.mul %276, %278 {shift = 0 : i8} : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
        %280 = tosa.mul %279, %19 {shift = 0 : i8} : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
        %281 = tosa.cast %280 : (tensor<1x1000xf32>) -> tensor<1x1000xi32>
        %282 = tosa.add %281, %18 : (tensor<1x1000xi32>, tensor<1x1xi32>) -> tensor<1x1000xi32>
        %283 = tosa.cast %282 : (tensor<1x1000xi32>) -> tensor<1x1000xi8>
        return %283 : tensor<1x1000xi8>
    }