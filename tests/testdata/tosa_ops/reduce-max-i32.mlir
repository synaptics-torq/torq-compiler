module {
    func.func @main(%arg0: tensor<1x1000xi32>) -> (tensor<1x1xi32>) {
        %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<1x1000xi32>) -> tensor<1x1xi32>
        return %0 : tensor<1x1xi32>
    }
}
