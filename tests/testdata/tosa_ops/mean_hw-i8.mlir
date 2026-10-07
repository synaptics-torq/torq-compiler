// TFLite MEAN over H and W (global average pooling) as imported for int8: a rescale to
// i32 that subtracts the input zero point, two reduce_sums, and a per-tensor rescale
// back to i8. It is raised to a whole-frame pooling sum that runs on the average pool
// kernel, with the input zero point folded into the bias.
module {
  func.func @main(%arg0: tensor<1x7x7x96xi8>) -> tensor<1x96xi8> {
    %unit_multiplier = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %unit_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mean_multiplier = "tosa.const"() <{values = dense<727255841> : tensor<1xi32>}> : () -> tensor<1xi32>
    %mean_shift = "tosa.const"() <{values = dense<35> : tensor<1xi8>}> : () -> tensor<1xi8>
    %input_zp = "tosa.const"() <{values = dense<-5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %acc_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %output_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_shape = tosa.const_shape {values = dense<[1, 96]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %widened = tosa.rescale %arg0, %unit_multiplier, %unit_shift, %input_zp, %acc_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x7x7x96xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x7x7x96xi32>
    %sum_h = tosa.reduce_sum %widened {axis = 1 : i32} : (tensor<1x7x7x96xi32>) -> tensor<1x1x7x96xi32>
    %sum = tosa.reduce_sum %sum_h {axis = 2 : i32} : (tensor<1x1x7x96xi32>) -> tensor<1x1x1x96xi32>
    %mean = tosa.rescale %sum, %mean_multiplier, %mean_shift, %acc_zp, %output_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x1x96xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x1x96xi8>
    %out = tosa.reshape %mean, %out_shape : (tensor<1x1x1x96xi8>, !tosa.shape<2>) -> tensor<1x96xi8>
    return %out : tensor<1x96xi8>
  }
}
