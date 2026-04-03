module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\0C\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00(\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\01\00\00\00\EB\03\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\00\0A\A5\C3\FB\E9m\C2\A2\02\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.18.0\00\00", min_runtime_version = "1.14.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x8x8x1xi8> {tf_saved_model.index_path = ["input_1"]}, %arg1: tensor<1x8x8x1xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x8x8x1xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0,serving_default_input_0:0", outputs = "PartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<50> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<1101864312> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<2144326152> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = tosa.rescale %arg1, %5, %6, %7, %8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x8x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x1xi32>
    %10 = tosa.rescale %arg0, %5, %4, %7, %8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x8x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x1xi32>
    %11 = tosa.rescale %10, %3, %2, %8, %8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x8x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x8x1xi32>
    %12 = tosa.add %9, %11 : (tensor<1x8x8x1xi32>, tensor<1x8x8x1xi32>) -> tensor<1x8x8x1xi32>
    %13 = tosa.rescale %12, %1, %0, %8, %7 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x8x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x1xi8>
    return %13 : tensor<1x8x8x1xi8>
  }
}

