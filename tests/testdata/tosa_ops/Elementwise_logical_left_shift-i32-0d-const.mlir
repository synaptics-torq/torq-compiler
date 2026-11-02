// A 0-d shift of constants: its linalg.generic reads them as tensor<1xi32> and writes a 0-d
// result, which torq_hl.elementwiseshift needs at a single shape. Not rescaled, so the
// broadcast pass does not compute it at compile time (see sub-rescaled-shifted-scalar-i16).
func.func @main(%arg0: tensor<1x6x8x4xi32>) -> tensor<1x6x8x4xi32> {
  %c = "tosa.const"() <{values = dense<32767> : tensor<i32>}> : () -> tensor<i32>
  %s = "tosa.const"() <{values = dense<3> : tensor<i32>}> : () -> tensor<i32>
  %sh = tosa.const_shape {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
  %l = tosa.logical_left_shift %c, %s : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %r = tosa.reshape %l, %sh : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
  %o = tosa.sub %arg0, %r : (tensor<1x6x8x4xi32>, tensor<1x1x1x1xi32>) -> tensor<1x6x8x4xi32>
  return %o : tensor<1x6x8x4xi32>
}
