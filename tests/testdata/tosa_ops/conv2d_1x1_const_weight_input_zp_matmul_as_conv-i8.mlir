// Matmul-as-conv regression (#1828): a TFLite CONV2D / FULLY_CONNECTED with a
// CONSTANT weight and a non-zero input zero-point lowers to a 1x1 tosa.conv2d
// (H=W=1). TosaToLinalg emits an input-zero-point correction as a broadcast
// linalg.generic whose input map is (d1,d2,d3) over a [1,1,C] tensor.
//
// raise-to-matmul intentionally skips constant-weight filters (only activation x
// activation 1x1 convs are raised), so this conv reaches ConvertNhwcOpToNchw.
// There a channel-only d3->d1 relabel collapsed the correction map to (d1,d2,d1)
// -- a non-injective map -- and compilation failed with
//   'linalg.generic' op inferred input/output operand #0 has shape's dimension
//   #2 to be 1, but found C
//
// The fix applies the full NHWC->NCHW loop-dim permutation (d1->d2, d2->d3,
// d3->d1) so the correction map relabels to (d2,d3,d1) -- still injective -- and
// the cluster converts to NCHW, fusing into a single torq_hl.conv2d. This case
// must compile and match the CPU reference.
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<8x16xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<8x32xi8> {ml_program.identifier = "StatefulPartitionedCall_0:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %in_shape = tosa.const_shape  {values = dense<[8, 1, 1, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %out_shape = tosa.const_shape  {values = dense<[8, 32]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %weight = "tosa.const"() <{values = dense<3> : tensor<32x1x1x16xi8>}> : () -> tensor<32x1x1x16xi8>
    %bias = "tosa.const"() <{values = dense<0> : tensor<32xi32>}> : () -> tensor<32xi32>
    %izp = "tosa.const"() <{values = dense<7> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mult = "tosa.const"() <{values = dense<1685431271> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %rin_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %rout_zp = "tosa.const"() <{values = dense<-5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.reshape %arg0, %in_shape : (tensor<8x16xi8>, !tosa.shape<4>) -> tensor<8x1x1x16xi8>
    %1 = tosa.conv2d %0, %weight, %bias, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<8x1x1x16xi8>, tensor<32x1x1x16xi8>, tensor<32xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<8x1x1x32xi32>
    %2 = tosa.rescale %1, %mult, %shift, %rin_zp, %rout_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<8x1x1x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<8x1x1x32xi8>
    %3 = tosa.reshape %2, %out_shape : (tensor<8x1x1x32xi8>, !tosa.shape<2>) -> tensor<8x32xi8>
    return %3 : tensor<8x32xi8>
  }
}
