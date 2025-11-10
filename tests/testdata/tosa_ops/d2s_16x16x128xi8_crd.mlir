
module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\10\00\00\00\00\00\00\00\08\00\0C\00\08\00\04\00\08\00\00\00\10\00\00\00,\00\00\00\08\00\0C\00\08\00\07\00\08\00\00\00\00\00\00\01\04\00\00\00\01\00\00\00\EB\03\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\001JW\A0w\9A\0Cn\03\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.18.0\00\00", min_runtime_version = "2.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x16x16x128xi8>) -> tensor<1x32x32x32xi8> attributes {tf.entry_function = {inputs = "input", outputs = "Identity"}} {
    %0 = "tosa.const"() <{value = dense<[0, 1, 4, 2, 5, 3]> : tensor<6xi32>}> : () -> tensor<6xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 16, 16, 32, 2, 2>} : (tensor<1x16x16x128xi8>) -> tensor<1x16x16x32x2x2xi8>
    %2 = tosa.transpose %1, %0 : (tensor<1x16x16x32x2x2xi8>, tensor<6xi32>) -> tensor<1x16x2x16x2x32xi8>
    %3 = tosa.reshape %2 {new_shape = array<i64: 1, 32, 32, 32>} : (tensor<1x16x2x16x2x32xi8>) -> tensor<1x32x32x32xi8>
    return %3 : tensor<1x32x32x32xi8>
  }
}

