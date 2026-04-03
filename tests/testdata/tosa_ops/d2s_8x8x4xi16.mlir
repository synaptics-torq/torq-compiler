module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\10\00\00\00\00\00\00\00\08\00\0C\00\08\00\04\00\08\00\00\00\10\00\00\00,\00\00\00\08\00\0C\00\08\00\07\00\08\00\00\00\00\00\00\01\04\00\00\00\01\00\00\00\EB\03\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\001JW\A0w\9A\0Cn\03\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.18.0\00\00", min_runtime_version = "2.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x8x8x4xi16>) -> tensor<1x16x16x1xi16> attributes {tf.entry_function = {inputs = "input", outputs = "Identity"}} {
    %0 = tosa.const_shape  {values = dense<[1, 8, 8, 2, 2, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %1 = tosa.reshape %arg0, %0 : (tensor<1x8x8x4xi16>, !tosa.shape<6>) -> tensor<1x8x8x2x2x1xi16>
    %2 = tosa.transpose %1 {perms = array<i32: 0, 1, 3, 2, 4, 5>} : (tensor<1x8x8x2x2x1xi16>) -> tensor<1x8x2x8x2x1xi16>
    %3 = tosa.const_shape  {values = dense<[1, 16, 16, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.reshape %2, %3 : (tensor<1x8x2x8x2x1xi16>, !tosa.shape<4>) -> tensor<1x16x16x1xi16>
    return %4 : tensor<1x16x16x1xi16>
  }
}

