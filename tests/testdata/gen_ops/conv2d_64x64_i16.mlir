module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\0C\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00(\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\01\00\00\00\EC\03\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\00\1B4\DB\19<('\ED\02\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.18.0\00\00", min_runtime_version = "1.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x64x64x1xi16> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x32x32x1xi16> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_input_0:0", outputs = "StatefulPartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<22> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<27016> : tensor<1xi16>}> : () -> tensor<1xi16>
    %2 = "tosa.const"() <{values = dense<[[[[38], [-13], [-94]], [[-99], [-56], [74]], [[127], [-106], [-22]]]]> : tensor<1x3x3x1xi8>}> : () -> tensor<1x3x3x1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi48>}> : () -> tensor<1xi48>
    %4 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %5 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %6 = tosa.conv2d %arg0, %2, %3, %4, %5 {acc_type = i48, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x64x64x1xi16>, tensor<1x3x3x1xi8>, tensor<1xi48>, tensor<1xi16>, tensor<1xi8>) -> tensor<1x32x32x1xi48>
    %7 = tosa.rescale %6, %1, %0, %3, %4 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = false} : (tensor<1x32x32x1xi48>, tensor<1xi16>, tensor<1xi8>, tensor<1xi48>, tensor<1xi16>) -> tensor<1x32x32x1xi16>
    %8 = tosa.clamp %7 {max_val = 32767 : i16, min_val = 0 : i16} : (tensor<1x32x32x1xi16>) -> tensor<1x32x32x1xi16>
    return %8 : tensor<1x32x32x1xi16>
  }
}

