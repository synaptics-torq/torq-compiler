// Matmul-as-conv regression: a TFLite FULLY_CONNECTED / batch-matmul lowered by
// tosa-converter-for-tflite to a 1x1 tosa.conv2d with a SIZE-1 zero bias
// (dense<0> : tensor<1xi32>) and two activation inputs (no constant weight).
//
// This exercises ConvertNhwcOpToNchw's handling of the bias-broadcast indexing
// map (which arrives as (d1) for the size-1 bias) and the weight-zero-point
// reduction generic. Before the fix this failed to compile with
//   'linalg.generic' op inferred input/output operand #1 has shape's dimension
//   #1 to be 1, but found 384
// See useful_agent_findings/tflite_fullyconnected_issue.md.
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<384x32xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<384x32xi8> {ml_program.identifier = "serving_default_input_1:0", tf_saved_model.index_path = ["input_1"]}) -> (tensor<384x384xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.const_shape  {values = dense<384> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<1685431271> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "tosa.const"() <{values = dense<-67> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.const"() <{values = dense<-16> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.const"() <{values = dense<-2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %6 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = tosa.const_shape  {values = dense<[384, 1, 1, 32]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = tosa.reshape %arg0, %7 : (tensor<384x32xi8>, !tosa.shape<4>) -> tensor<384x1x1x32xi8>
    %9 = tosa.reshape %arg1, %7 : (tensor<384x32xi8>, !tosa.shape<4>) -> tensor<384x1x1x32xi8>
    %10 = tosa.conv2d %8, %9, %6, %5, %4 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<384x1x1x32xi8>, tensor<384x1x1x32xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<384x1x1x384xi32>
    %11 = tosa.rescale %10, %2, %1, %6, %3 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<384x1x1x384xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<384x1x1x384xi8>
    %12 = tosa.reshape %11, %0 : (tensor<384x1x1x384xi8>, !tosa.shape<2>) -> tensor<384x384xi8>
    return %12 : tensor<384x384xi8>
  }
}
