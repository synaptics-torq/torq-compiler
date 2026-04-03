module {
  func.func @main(%arg0: tensor<1x1000xi8>) -> tensor<1x1000xi8> {
    %0 = "tosa.const"() <{values = dense<-128> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
    %1 = "tosa.const"() <{values = dense<2.560000e+02> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    %2 = "tosa.const"() <{values = dense<0.0673685893> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    %3 = "tosa.const"() <{values = dense<-4.800000e+01> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    %cst = arith.constant dense<0> : tensor<1xi8>
    %4 = tosa.cast %arg0 : (tensor<1x1000xi8>) -> tensor<1x1000xf32>
    %5 = tosa.sub %4, %3 : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
    %6 = tosa.mul %5, %2, %cst : (tensor<1x1000xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x1000xf32>
    %7 = tosa.reduce_max %6 {axis = 1 : i32} : (tensor<1x1000xf32>) -> tensor<1x1xf32>
    %8 = tosa.sub %6, %7 : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
    %9 = tosa.exp %8 : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %10 = tosa.reduce_sum %9 {axis = 1 : i32} : (tensor<1x1000xf32>) -> tensor<1x1xf32>
    %11 = tosa.reciprocal %10 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %12 = tosa.mul %9, %11, %cst : (tensor<1x1000xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x1000xf32>
    %13 = tosa.mul %12, %1, %cst : (tensor<1x1000xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x1000xf32>
    %14 = tosa.cast %13 : (tensor<1x1000xf32>) -> tensor<1x1000xi32>
    %15 = tosa.add %14, %0 : (tensor<1x1000xi32>, tensor<1x1xi32>) -> tensor<1x1000xi32>
    %16 = tosa.cast %15 : (tensor<1x1000xi32>) -> tensor<1x1000xi8>
    return %16 : tensor<1x1000xi8>
  }
}

