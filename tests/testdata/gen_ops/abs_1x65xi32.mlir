module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\0C\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00(\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\01\00\00\00\EA\03\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\00\C2\029C\AD\C3:~\02\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.18.0\00\00", min_runtime_version = "2.12.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<?x1x65xi32> {tf_saved_model.index_path = ["keras_tensor_8"]}) -> (tensor<?x1x65xi32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_keras_tensor_8:0", outputs = "PartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.abs %arg0 : (tensor<?x1x65xi32>) -> tensor<?x1x65xi32>
    return %0 : tensor<?x1x65xi32>
  }
}

