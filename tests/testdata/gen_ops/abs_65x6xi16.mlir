module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\0C\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00(\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\01\00\00\00\EC\03\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\00F\F3\FD\8F,\B23\DC\02\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.18.0\00\00", min_runtime_version = "1.13.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<?x65x6xi16> {tf_saved_model.index_path = ["keras_tensor_9"]}) -> (tensor<?x65x6xi16> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_keras_tensor_9:0", outputs = "PartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.abs %arg0 : (tensor<?x65x6xi16>) -> tensor<?x65x6xi16>
    return %0 : tensor<?x65x6xi16>
  }
}

