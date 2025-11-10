func.func @main(%201 : tensor<1x21x1024xi32> ) -> tensor<1x21x1024xi32> {
    %6 = "tosa.const"() <{value = dense<32767> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
    %203 = tosa.add %201, %6 : (tensor<1x21x1024xi32>, tensor<1x1x1xi32>) -> tensor<1x21x1024xi32>
    return %203 : tensor<1x21x1024xi32>
}
