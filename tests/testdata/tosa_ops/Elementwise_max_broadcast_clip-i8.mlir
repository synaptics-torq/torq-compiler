module attributes {tfl.description = "Extracted single layer", tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x4x384x384xi8>) -> tensor<1x4x384x384xi8> attributes {tf.entry_function = {inputs = "bert/encoder/layer_0/attention/self/clip_by_value/Minimum1", outputs = "bert/encoder/layer_0/attention/self/clip_by_value"}} {
    %0 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = "tosa.const"() <{values = dense<-128> : tensor<i8>}> : () -> tensor<i8>
    %2 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = tosa.rescale %arg0, %2, %3, %4, %5 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x4x384x384xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x4x384x384xi32>
    %7 = tosa.rescale %1, %2, %3, %4, %5 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<i8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<i32>
    %8 = tosa.reshape %7, %0 : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
    %9 = tosa.maximum %6, %8 : (tensor<1x4x384x384xi32>, tensor<1x1x1x1xi32>) -> tensor<1x4x384x384xi32>
    %10 = tosa.rescale %9, %2, %3, %5, %4 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x4x384x384xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x4x384x384xi8>
    return %10 : tensor<1x4x384x384xi8>
  }
}
