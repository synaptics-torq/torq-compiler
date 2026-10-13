// The resnet50 conv5_block2_1_conv layer (1x1 conv, 2048 to 512 channels, 7x7) with splat
// constants. On SL2610, its 4 tiles of 128 output channels relocate a buffer. The only tile
// without relocation within the tile budget cuts the channels to 103, which leaves a partial
// channel vector in every tile.
module {
  func.func @main(%arg0: tensor<1x7x7x2048xi8>) -> tensor<1x7x7x512xi8> {
    %0 = "tosa.const"() <{values = dense<44> : tensor<512xi8>}> : () -> tensor<512xi8>
    %1 = "tosa.const"() <{values = dense<1073741824> : tensor<512xi32>}> : () -> tensor<512xi32>
    %2 = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "tosa.const"() <{values = dense<1> : tensor<512x1x1x2048xi8>}> : () -> tensor<512x1x1x2048xi8>
    %5 = "tosa.const"() <{values = dense<0> : tensor<512xi32>}> : () -> tensor<512xi32>
    %6 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = tosa.conv2d %arg0, %4, %5, %6, %7 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x2048xi8>, tensor<512x1x1x2048xi8>, tensor<512xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x7x7x512xi32>
    %9 = tosa.rescale %8, %1, %0, %3, %2 {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x7x7x512xi32>, tensor<512xi32>, tensor<512xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x7x7x512xi8>
    return %9 : tensor<1x7x7x512xi8>
  }
}
