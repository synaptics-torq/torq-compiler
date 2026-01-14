module {
    func.func @main(%arg0: tensor<1x1000xf32>) -> (tensor<1x1xf32>) {
        %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<1x1000xf32>) -> tensor<1x1xf32>
        return %0 : tensor<1x1xf32>
    }
}
