// 6x6 stride-2 depthwise conv with no padding (synaptics-torq/torq-compiler-dev#2717)
module {
  func.func @main(%arg0: tensor<1x12x12x8xi8>) -> tensor<1x4x4x8xi8> {
    %w = "tosa.const"() <{values = dense<"0x1FFBDC3C7A3510B77B1844D6F364BB93697C020D6AC8910C4BC20A8D396C9AB13A13DFF4D22DC2912243673CAF2A7344EDCF3134A767BB5195A7924C13D178E5A2D992F31C15C8CB07780EAB85E570C673DFA738131EE8DCEBD3AD7EA3B82F12A3B5CE29F5B79F0624E68F4B76A977DA0B2B3D1BF385AE227D5F2B5CD6987E2BA2D39A18FFDE6A0D693C8D8EF241A5E3D25ECF3B97371EF8BD63D0576F0516FA361EFF5CF69FD7BF454A0060131DCDFAD4ECB752B2350A38079C7C1DED411B8BAB741CD7C9DD1BFB1F4C60762A81E2E3C907EC8C0E550FE174760D98E6DB7B148F8EAE4E8312ED876C74B5744E4E5645CA290CA2205094D02DE6AFF6748E5FF8A9D8F113269370FA5B0497A80489C354EEA00B909499839C1699E4FBF025FB64"> : tensor<6x6x8x1xi8>}> : () -> tensor<6x6x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0x85FDFFFF55030000AF030000000000001B030000E4FFFFFF990300006EFCFFFF"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0xEBCCA5663D8DB3422F63F8602D028C78CD3F4B753170AE4DDF15C069E647B14A"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x12x12x8xi8>, tensor<6x6x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x4x4x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x4x4x8xi8>
    return %1 : tensor<1x4x4x8xi8>
  }
}
