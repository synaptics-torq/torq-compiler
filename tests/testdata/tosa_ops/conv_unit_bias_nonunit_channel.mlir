module attributes {tfl.description = "Reduced repro: 1xi32 bias into non-unit channel dim", tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x4x2xi8>) -> tensor<1x4x8xi8> attributes {tf.entry_function = {inputs = "input", outputs = "output"}} {
    %out_shape = tosa.const_shape {values = dense<[1, 4, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %w_shape = tosa.const_shape {values = dense<[8, 1, 1, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %in_shape = tosa.const_shape {values = dense<[4, 1, 1, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>

    // Single-element bias: this is the operand that gets mis-promoted.
    %bias = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>

    // conv2d zero points
    %in_zp = "tosa.const"() <{values = dense<-100> : tensor<1xi8>}> : () -> tensor<1xi8>
    %w_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>

    // rescale operands (per_channel = false -> also single-element broadcasts)
    %mult = "tosa.const"() <{values = dense<2062378828> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<39> : tensor<1xi8>}> : () -> tensor<1xi8>
    %rs_in_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %rs_out_zp = "tosa.const"() <{values = dense<4> : tensor<1xi8>}> : () -> tensor<1xi8>

    %weights = "tosa.const"() <{values = dense<[[1, -2], [0, 3], [-1, 1], [2, 0], [1, 1], [-3, 2], [0, -1], [2, 2]]> : tensor<8x2xi8>}> : () -> tensor<8x2xi8>

    %in_4d = tosa.reshape %arg0, %in_shape : (tensor<1x4x2xi8>, !tosa.shape<4>) -> tensor<4x1x1x2xi8>
    %w_4d = tosa.reshape %weights, %w_shape : (tensor<8x2xi8>, !tosa.shape<4>) -> tensor<8x1x1x2xi8>

    %conv = tosa.conv2d %in_4d, %w_4d, %bias, %in_zp, %w_zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<4x1x1x2xi8>, tensor<8x1x1x2xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x1x1x8xi32>

    %rescaled = tosa.rescale %conv, %mult, %shift, %rs_in_zp, %rs_out_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<4x1x1x8xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<4x1x1x8xi8>

    %result = tosa.reshape %rescaled, %out_shape : (tensor<4x1x1x8xi8>, !tosa.shape<3>) -> tensor<1x4x8xi8>
    return %result : tensor<1x4x8xi8>
  }
}
