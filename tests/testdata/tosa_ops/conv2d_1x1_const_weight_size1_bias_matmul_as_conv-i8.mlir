// Matmul-as-conv regression: a TFLite CONV2D / FULLY_CONNECTED with a CONSTANT
// weight and a SIZE-1 zero bias (dense<0> : tensor<1xi32>) lowers to a 1x1
// tosa.conv2d (H=W=1). raise-to-matmul intentionally skips constant-weight
// filters (only activation x activation 1x1 convs are raised), so this conv
// reaches ConvertNhwcOpToNchw.
//
// PromoteScalarsTo1D rewrites the scalar bias to a rank-1 tensor<1xi32> pinned to
// the first unit output dim, giving the bias-broadcast indexing map (d1) over a
// [.,1,1,.] output -- i.e. pinned to a SPATIAL unit dim, not the channel dim.
// The NHWC->NCHW relabel must move that spatial dim too (d1->d2), because the
// spatial axes shift under the layout change. A relabel that only remapped the
// channel (d3->d1) left (d1) unchanged, so after the output became [.,C,1,1] the
// size-1 source sat on loop dim d1 (extent C) and failed verification with
//   'linalg.generic' op inferred input/output operand #1 has shape's dimension
//   #1 to be 1, but found 32
//
// This complements conv2d_1x1_const_weight_input_zp_matmul_as_conv-i8.mlir, whose
// zero-point-correction broadcast exercises the non-injective (d1,d2,d3) map. Both
// convert cleanly once the full NHWC->NCHW loop-dim permutation is applied, and
// the whole conv+bias+rescale cluster fuses into a single torq_hl.conv2d. This
// case must compile and match the CPU reference.
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<8x16xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<8x32xi8> {ml_program.identifier = "StatefulPartitionedCall_0:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %in_shape = tosa.const_shape  {values = dense<[8, 1, 1, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %out_shape = tosa.const_shape  {values = dense<[8, 32]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %weight = "tosa.const"() <{values = dense<3> : tensor<32x1x1x16xi8>}> : () -> tensor<32x1x1x16xi8>
    %bias = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %izp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mult = "tosa.const"() <{values = dense<1685431271> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %rin_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %rout_zp = "tosa.const"() <{values = dense<-5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.reshape %arg0, %in_shape : (tensor<8x16xi8>, !tosa.shape<4>) -> tensor<8x1x1x16xi8>
    %1 = tosa.conv2d %0, %weight, %bias, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<8x1x1x16xi8>, tensor<32x1x1x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<8x1x1x32xi32>
    %2 = tosa.rescale %1, %mult, %shift, %rin_zp, %rout_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<8x1x1x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<8x1x1x32xi8>
    %3 = tosa.reshape %2, %out_shape : (tensor<8x1x1x32xi8>, !tosa.shape<2>) -> tensor<8x32xi8>
    return %3 : tensor<8x32xi8>
  }
}
