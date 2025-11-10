module {
func.func @main(%arg: tensor<64x64x3xi8>) -> (tensor<66x66x3xi8>) {
    %c-1_i8 = arith.constant -1 : i8
    
    %padded = tensor.pad %arg low[1, 1, 0] high[1, 1, 0] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index):
        tensor.yield %c-1_i8 : i8
    } : tensor<64x64x3xi8> to tensor<66x66x3xi8>

    return %padded : tensor<66x66x3xi8>
}
}