// Grouped 1x1 conv (H=1, W>1) whose CONSTANT filter reaches the conv through a
// tosa.slice -> tensor.extract_slice. tracesToConstantWeight must see through
// extract_slice; otherwise the conv is wrongly raised to a matmul that collapses
// [N,H,W,C]->[N,C] (valid only for H=W=1) and fails verification for W>1. Weight
// is non-splat so the slice stays a real extract_slice.
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<3x2xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<3x2xi8> {ml_program.identifier = "StatefulPartitionedCall_0:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %in_shape = tosa.const_shape  {values = dense<[1, 1, 3, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %out_shape = tosa.const_shape  {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %w_start = tosa.const_shape  {values = dense<[2, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %w_size = tosa.const_shape  {values = dense<[2, 1, 1, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %w_full = "tosa.const"() <{values = dense<[[[[1, 2]]], [[[-3, 4]]], [[[5, -6]]], [[[7, 8]]]]> : tensor<4x1x1x2xi8>}> : () -> tensor<4x1x1x2xi8>
    %bias = "tosa.const"() <{values = dense<0> : tensor<2xi32>}> : () -> tensor<2xi32>
    %izp = "tosa.const"() <{values = dense<-18> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<36> : tensor<1xi8>}> : () -> tensor<1xi8>
    %rin_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %rout_zp = "tosa.const"() <{values = dense<4> : tensor<1xi8>}> : () -> tensor<1xi8>
    %w = tosa.slice %w_full, %w_start, %w_size : (tensor<4x1x1x2xi8>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x1x1x2xi8>
    %0 = tosa.reshape %arg0, %in_shape : (tensor<3x2xi8>, !tosa.shape<4>) -> tensor<1x1x3x2xi8>
    %1 = tosa.conv2d %0, %w, %bias, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x3x2xi8>, tensor<2x1x1x2xi8>, tensor<2xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x3x2xi32>
    %2 = tosa.rescale %1, %mult, %shift, %rin_zp, %rout_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x3x2xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x3x2xi8>
    %3 = tosa.reshape %2, %out_shape : (tensor<1x1x3x2xi8>, !tosa.shape<2>) -> tensor<3x2xi8>
    return %3 : tensor<3x2xi8>
  }
}
