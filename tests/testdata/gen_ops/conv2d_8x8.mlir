module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\0C\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00(\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\01\00\00\00\EB\03\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\00\94\BA\D2\8F\ABo3\EA\02\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.18.0\00\00", min_runtime_version = "1.14.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x8x8x1xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x8x8x1xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_input_0:0", outputs = "StatefulPartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<39> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<2003370132> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "tosa.const"() <{values = dense<[[[[119], [23], [-114]], [[115], [-127], [125]], [[7], [-35], [-13]]]]> : tensor<1x3x3x1xi8>}> : () -> tensor<1x3x3x1xi8>
    %4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %6 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = tosa.conv2d %arg0, %3, %4, %5, %6 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x1xi8>, tensor<1x3x3x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x1xi32>
    %8 = tosa.rescale %7, %1, %0, %2, %5 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x8x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x1xi8>
    %9 = tosa.clamp %8 {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x8x8x1xi8>) -> tensor<1x8x8x1xi8>
    return %9 : tensor<1x8x8x1xi8>
  }
}

